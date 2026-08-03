//! Predictive prefetch for SSD-MoE: Markov transition model + locality monitor.
//!
//! After every MoE forward pass the engine feeds the routing decision to both
//! predictors and receives a ranked set of expert IDs to speculatively prefetch
//! from SSD. All prefetches are non-evicting.

mod locality;
mod markov;

use locality::LocalityMonitor;
use markov::MarkovPredictor;

/// Tiny rank-tie-break decay so earlier arm positions slightly outrank later ones.
const RANK_TIEBREAK_DECAY: f32 = 0.0001;

pub struct PredictiveLoader {
    markov: MarkovPredictor,
    fanout: usize,
}

impl PredictiveLoader {
    pub fn new(num_experts: usize, fanout: usize) -> Self {
        Self {
            markov: MarkovPredictor::new(num_experts),
            fanout,
        }
    }

    pub fn observe_and_predict(
        &mut self,
        prev_set: &[u32],
        next_set: &[u32],
        locality: Option<&LocalityMonitor>,
        locality_threshold: f64,
    ) -> Vec<u32> {
        self.markov.observe_step(prev_set, next_set);
        let locality_hot: Vec<u32> = locality
            .as_ref()
            .map(|loc| loc.hot_set(locality_threshold))
            .unwrap_or_default();
        let markov_preds = self.markov.predict_for_set(&[], next_set);
        let scored = self.combine_unified_arms(&markov_preds, &locality_hot);
        scored
            .into_iter()
            .map(|(id, _)| id)
            .take(self.fanout)
            .collect()
    }

    pub fn combine_unified_arms(
        &self,
        markov: &[(u32, f64)],
        locality: &[u32],
    ) -> Vec<(u32, f32)> {
        use std::collections::BTreeMap;
        let mut scores: BTreeMap<u32, f32> = BTreeMap::new();
        for (rank, &(id, p)) in markov.iter().enumerate() {
            let decay = 1.0 - RANK_TIEBREAK_DECAY * rank as f32;
            *scores.entry(id).or_insert(0.0) += markov::W_MARKOV * (p as f32) * decay;
        }
        for (rank, &id) in locality.iter().enumerate() {
            let decay = 1.0 - RANK_TIEBREAK_DECAY * rank as f32;
            *scores.entry(id).or_insert(0.0) += locality::W_LOCALITY * decay;
        }
        let mut out: Vec<(u32, f32)> = scores.into_iter().collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unified_combine_ranks_multi_arm_expert_highest() {
        let pl = PredictiveLoader::new(8, 4);
        let markov = vec![(1u32, 1.0f64)];
        let locality = vec![2u32];
        let out = pl.combine_unified_arms(&markov, &locality);
        assert_eq!(out[0].0, 1);
        assert_eq!(out[1].0, 2);
    }

    #[test]
    fn unified_combine_merges_same_id() {
        let pl = PredictiveLoader::new(8, 4);
        let markov = vec![(3u32, 0.8f64)];
        let locality = vec![3u32];
        let out = pl.combine_unified_arms(&markov, &locality);
        assert_eq!(out.len(), 1);
        let expected = markov::W_MARKOV * 0.8 + locality::W_LOCALITY;
        assert!((out[0].1 - expected).abs() < 0.01);
    }

    #[test]
    fn observe_and_predict_learns() {
        let mut pl = PredictiveLoader::new(8, 4);
        let loc = LocalityMonitor::new(8, 32);
        for _ in 0..50 {
            pl.observe_and_predict(&[0], &[1], Some(&loc), 0.10);
        }
        let preds = pl.observe_and_predict(&[1], &[1], Some(&loc), 0.10);
        assert!(!preds.is_empty());
        let hot = loc.hot_set(0.10);
    }
}
