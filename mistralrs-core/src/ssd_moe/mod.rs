//! SSD-streamed MoE expert backend.
//!
//! When the `ssd-moe` feature is enabled, this module provides an alternative
//! `MoEExpertsBackendImpl` variant that streams expert weights from SSD via
//! `O_DIRECT` pread(2), caches them in a configurable RAM LRU, and executes
//! SwiGLU FFN over the cached bytes. Optional predictors can speculatively
//! prefetch experts to hide SSD latency.
//!
//! Without the feature, this module compiles out — zero cost.

mod aligned_buffer;
mod backend;
pub(super) mod buffer_pool;
pub mod config;
mod dequant;
pub(super) mod expert_cache;
pub mod manifest;
pub mod predictors;
pub(super) mod storage;

#[cfg(all(target_os = "linux", feature = "io-uring"))]
mod io_uring;
#[cfg(all(target_os = "linux", feature = "io-uring"))]
pub use io_uring::IoUringStorage;

pub use backend::SsdMoeBackend;
pub use config::SsdMoeConfig;
