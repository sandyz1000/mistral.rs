//! Sliding-window locality monitor: tracks recently active experts to pin hot ones.

use std::collections::VecDeque;

pub const W_LOCALITY: f32 = 0.25;

pub struct LocalityMonitor {
    window: VecDeque<u32>,
    window_size: usize,
    counts: Vec<usize>,
    total: usize,
}

impl LocalityMonitor {
    pub fn new(num_experts: usize, window_size: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(window_size),
            window_size,
            counts: vec![0; num_experts],
            total: 0,
        }
    }

    pub fn observe_one(&mut self, id: u32) {
        if self.window.len() >= self.window_size {
            if let Some(old) = self.window.pop_front() {
                self.counts[old as usize] = self.counts[old as usize].saturating_sub(1);
                self.total = self.total.saturating_sub(1);
            }
        }
        self.window.push_back(id);
        self.counts[id as usize] += 1;
        self.total += 1;
    }

    pub fn observe(&mut self, ids: &[u32]) {
        for &id in ids {
            self.observe_one(id);
        }
    }

    pub fn hot_set(&self, pct_threshold: f64) -> Vec<u32> {
        if self.total == 0 {
            return Vec::new();
        }
        let min_count = (self.total as f64 * pct_threshold).ceil() as usize;
        let mut hot: Vec<u32> = self
            .counts
            .iter()
            .enumerate()
            .filter(|(_, &c)| c >= min_count)
            .map(|(id, _)| id as u32)
            .collect();
        hot.sort_unstable();
        hot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hot_set_threshold() {
        let mut m = LocalityMonitor::new(8, 16);
        for _ in 0..8 { m.observe_one(2); }
        for _ in 0..3 { m.observe_one(0); }
        for i in 4..9 { m.observe_one(i as u32 % 8); }
        let hot = m.hot_set(0.10);
        assert!(hot.contains(&2));
        assert!(hot.contains(&0));
    }

    #[test]
    fn sliding_window_evicts_old() {
        let mut m = LocalityMonitor::new(8, 4);
        for _ in 0..4 { m.observe_one(0); }
        for _ in 0..4 { m.observe_one(1); }
        let hot = m.hot_set(0.10);
        assert_eq!(hot, vec![1]);
    }
}
