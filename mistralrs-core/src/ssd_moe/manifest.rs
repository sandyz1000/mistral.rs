//! manifest.json reader — metadata about expert files produced by the
//! extraction tool.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Deserialize)]
pub struct Manifest {
    pub num_layers: usize,
    pub num_experts_per_layer: usize,
    pub layers: Vec<LayerInfo>,
    pub expert_map: BTreeMap<u32, ExpertEntry>,
}

#[derive(Deserialize)]
pub struct LayerInfo {
    #[allow(dead_code)]
    pub layer_idx: usize,
    pub num_experts: usize,
    #[allow(dead_code)]
    pub expert_ids: Vec<u32>,
}

#[derive(Deserialize, Clone)]
pub struct ExpertEntry {
    #[allow(dead_code)]
    pub layer_idx: usize,
    pub local_id: usize,
    pub file: String,
    pub dtype: String,
    pub d_model: usize,
    pub d_ff: usize,
    pub byte_size: u64,
}

impl Manifest {
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path.as_ref())?;
        let m: Self = serde_json::from_slice(&bytes)?;
        Ok(m)
    }

    /// Look up an expert by global id.
    pub fn lookup(&self, global_id: u32) -> Option<&ExpertEntry> {
        self.expert_map.get(&global_id)
    }
}
