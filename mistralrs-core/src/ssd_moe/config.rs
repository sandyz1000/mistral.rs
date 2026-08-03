/// Configuration for SSD-backed MoE expert streaming.
#[derive(Clone, Debug)]
pub struct SsdMoeConfig {
    pub expert_dir: std::path::PathBuf,
    pub cache_slots: usize,
    /// Number of experts to speculatively prefetch per token (0 = disabled).
    pub prefetch_fanout: usize,
    /// Enable the locality monitor (pins hot experts in cache).
    pub locality_enabled: bool,
}

impl SsdMoeConfig {
    pub fn new(expert_dir: std::path::PathBuf) -> Self {
        Self {
            expert_dir,
            cache_slots: 256,
            prefetch_fanout: 0,
            locality_enabled: false,
        }
    }

    pub fn with_cache_slots(mut self, n: usize) -> Self {
        self.cache_slots = n;
        self
    }

    pub fn with_prefetch(mut self, fanout: usize) -> Self {
        self.prefetch_fanout = fanout;
        self
    }

    pub fn with_locality(mut self, on: bool) -> Self {
        self.locality_enabled = on;
        self
    }
}
