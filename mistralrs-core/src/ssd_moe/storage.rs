//! Expert storage backends.
//!
//! [`ExpertStorage`] is the trait abstracting over how expert bytes are
//! fetched from disk. [`PreadStorage`] is the portable `pread(2)` + `O_DIRECT`
//! implementation. An io_uring backend can be added later implementing the
//! same trait.

use super::buffer_pool::PooledBuffer;
use super::manifest::Manifest;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(target_os = "linux")]
use std::os::unix::fs::OpenOptionsExt;

/// Abstract storage for expert weight files.
pub trait ExpertStorage: Send + Sync {
    /// Read the full raw bytes of `expert_id` into `buf`.
    ///
    /// `buf.len()` must equal the expected expert byte size (as known from the
    /// manifest).
    async fn read_expert(
        &self,
        expert_id: u32,
        buf: &mut PooledBuffer,
    ) -> io::Result<usize>;
}

/// Portable storage backend: one file per expert, opened lazily, read via
/// `pread(2)` with optional `O_DIRECT`.
pub struct PreadStorage {
    base_path: PathBuf,
    manifest: Manifest,
    expert_size: usize,
    use_direct_io: bool,
}

impl PreadStorage {
    pub fn new(base_path: PathBuf, manifest: Manifest) -> anyhow::Result<Self> {
        // Determine expert size from the first entry in the manifest.
        let expert_size = manifest
            .expert_map
            .values()
            .next()
            .map(|e| e.byte_size as usize)
            .unwrap_or(0);
        Ok(Self {
            base_path,
            manifest,
            expert_size,
            use_direct_io: true,
        })
    }

    pub fn set_direct_io(&mut self, on: bool) {
        self.use_direct_io = on;
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    pub fn expert_path(&self, expert_id: u32) -> PathBuf {
        match self.manifest.lookup(expert_id) {
            Some(entry) => self.base_path.join(&entry.file),
            None => self
                .base_path
                .join(format!("expert_{expert_id}.bin")),
        }
    }

    fn open_expert_file(&self, path: &Path) -> io::Result<Arc<File>> {
        let mut opts = OpenOptions::new();
        opts.read(true);
        #[cfg(target_os = "linux")]
        if self.use_direct_io {
            opts.custom_flags(libc::O_DIRECT);
        }
        let file = opts.open(path)?;
        Ok(Arc::new(file))
    }

    /// Read an expert's full content into `buf` via `pread`.
    ///
    /// Runs synchronously via `tokio::task::block_in_place` so the Tokio
    /// runtime stays responsive during the I/O.
    pub async fn read_expert_sync(
        &self,
        expert_id: u32,
        buf: &mut PooledBuffer,
    ) -> io::Result<usize> {
        let path = self.expert_path(expert_id);
        let file = self.open_expert_file(&path)?;
        let n = buf.len().min(self.expert_size);

        let result = tokio::task::block_in_place(|| file.read_at(buf.as_mut_slice(), 0));

        result.map(|rd| {
            if rd < n {
                tracing::warn!(
                    expert_id,
                    expected = n,
                    got = rd,
                    "short read on expert file"
                );
            }
            rd
        })
    }
}

impl ExpertStorage for PreadStorage {
    async fn read_expert(
        &self,
        expert_id: u32,
        buf: &mut PooledBuffer,
    ) -> io::Result<usize> {
        self.read_expert_sync(expert_id, buf).await
    }
}
