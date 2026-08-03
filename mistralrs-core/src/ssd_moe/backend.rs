//! SSD-MoE backend: wires cache + storage + dequant into an
//! `MoEExpertsBackendImpl` variant.

use super::dequant::ExpertDtype;
use super::buffer_pool::BufferPool;
use super::expert_cache::{ExpertCache, ExpertResident};
use super::manifest::Manifest;
use super::storage::PreadStorage;
use crate::layers::Activation;
use candle_core::{Device, Result, Tensor};
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::sync::Arc;

pub struct SsdMoeBackend {
    storage: PreadStorage,
    cache: Arc<ExpertCache>,
    pool: BufferPool,
    expert_dtype: ExpertDtype,
    d_model: usize,
    d_ff: usize,
    num_experts_per_tok: usize,
    #[allow(dead_code)]
    act: Activation,
}

impl SsdMoeBackend {
    pub fn new(
        expert_dir: PathBuf,
        manifest: Manifest,
        cache_slots: usize,
        d_model: usize,
        d_ff: usize,
        num_experts_per_tok: usize,
        act: Activation,
    ) -> anyhow::Result<Self> {
        let entry = manifest
            .expert_map
            .values()
            .next()
            .ok_or_else(|| anyhow::anyhow!("manifest has no expert entries"))?;
        let expert_dtype = ExpertDtype::from_manifest_str(&entry.dtype);
        let expert_size = expert_dtype.expert_byte_size(d_ff, d_model);
        let pool = BufferPool::new(cache_slots, expert_size, 4096);
        Ok(Self {
            storage: PreadStorage::new(expert_dir, manifest)?,
            cache: Arc::new(ExpertCache::new(cache_slots)),
            pool,
            expert_dtype,
            d_model,
            d_ff,
            num_experts_per_tok,
            act,
        })
    }

    pub fn manifest(&self) -> &Manifest {
        self.storage.manifest()
    }

    pub fn cache(&self) -> &Arc<ExpertCache> {
        &self.cache
    }

    /// Forward pass for the SSD-backed MoE layer.
    pub fn forward(
        &self,
        xs: &Tensor,
        topk_weights: &Tensor,
        topk_ids: &Tensor,
    ) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden_dim))?;
        let num_tokens = xs_flat.dims()[0];
        let dtype = xs.dtype();
        let device = xs.device();

        let mut token_outputs = Vec::with_capacity(num_tokens);

        for token_idx in 0..num_tokens {
            let token_hidden = xs_flat.get(token_idx)?;
            let token_weights = topk_weights.get(token_idx)?;
            let token_ids = topk_ids.get(token_idx)?;

            let n_experts = token_weights.dims()[0].min(self.num_experts_per_tok);
            let mut token_out = Tensor::zeros((hidden_dim,), dtype, device)?;

            for e in 0..n_experts {
                let weight = token_weights.get(e)?.to_scalar::<f32>()?;
                let expert_id = token_ids.get(e)?.to_scalar::<u32>()?;

                let expert_output = self.fetch_and_compute(expert_id, &token_hidden)?;
                token_out = (token_out + expert_output.affine(weight as f64, 0.0)?)?;
            }

            token_outputs.push(token_out);
        }

        let stacked = Tensor::stack(&token_outputs.iter().collect::<Vec<_>>(), 0)?;
        stacked.reshape((b_size, seq_len, hidden_dim))
    }

    /// Fetch expert bytes (cache hit or SSD read), dequantize, then SwiGLU FFN.
    fn fetch_and_compute(&self, expert_id: u32, hidden: &Tensor) -> Result<Tensor> {
        let resident = match self.cache.get(expert_id) {
            Some(r) => r,
            None => self.load_expert(expert_id)?,
        };

        let data = resident.data();
        let n_ff = self.d_ff;
        let n_model = self.d_model;
        let gate_sz = self.expert_dtype.proj_byte_size(n_ff, n_model);
        let up_sz = self.expert_dtype.proj_byte_size(n_ff, n_model);
        let down_sz = self.expert_dtype.proj_byte_size(n_model, n_ff);

        assert!(
            data.len() >= gate_sz + up_sz + down_sz,
            "expert data too short: got {} need {}",
            data.len(),
            gate_sz + up_sz + down_sz
        );

        let device = hidden.device();
        let gate = self.dequant_and_load(&data[..gate_sz], n_ff, n_model, device)?;
        let up = self.dequant_and_load(&data[gate_sz..gate_sz + up_sz], n_ff, n_model, device)?;
        let down = self.dequant_and_load(
            &data[gate_sz + up_sz..gate_sz + up_sz + down_sz],
            n_model,
            n_ff,
            device,
        )?;

        let gate_out = gate.matmul(&hidden.unsqueeze(1)?.t()?)?;
        let up_out = up.matmul(&hidden.unsqueeze(1)?.t()?)?;
        let act = candle_nn::ops::silu(&gate_out)?;
        let interm = act.broadcast_mul(&up_out)?;
        let result = down.t()?.matmul(&interm)?;
        Ok(result.squeeze(1)?.squeeze(0)?)
    }

    fn dequant_and_load(
        &self,
        bytes: &[u8],
        rows: usize,
        cols: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let f32s = self.expert_dtype.dequant(bytes, rows, cols);
        Tensor::from_vec(f32s, (rows, cols), device)
    }

    fn load_expert(&self, expert_id: u32) -> Result<Arc<ExpertResident>> {
        let mut buf = match self.pool.try_acquire() {
            Some(b) => b,
            None => match self.cache.evict_lru() {
                Some(_evicted) => self
                    .pool
                    .try_acquire()
                    .expect("buffer should be free after eviction"),
                None => candle_core::bail!("cache full and all experts pinned"),
            },
        };

        let path = self.storage.expert_path(expert_id);
        let file = std::fs::File::open(&path)
            .map_err(|e| candle_core::Error::Msg(format!("open expert {expert_id}: {e}")))?;
        let n = file
            .read_at(buf.as_mut_slice(), 0)
            .map_err(|e| candle_core::Error::Msg(format!("read expert {expert_id}: {e}")))?;

        if n < buf.len() {
            candle_core::bail!(
                "short read on expert {expert_id}: got {n} expected {}",
                buf.len()
            );
        }

        let resident = Arc::new(ExpertResident::new(expert_id, buf));
        let _ = self.cache.insert(resident.clone());
        Ok(resident)
    }
}
