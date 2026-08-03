//! Online-trained 2nd-order Markov transition model over expert routing decisions.

use std::collections::HashMap;

const LAPLACE_ALPHA: f64 = 1.0;
const MIN_2ND_ORDER_OBS: u64 = 10;

pub const W_MARKOV: f32 = 0.33;

pub struct MarkovPredictor {
    num_experts: usize,
    trans1: Vec<HashMap<u32, u64>>,
    total1: Vec<u64>,
    trans2: HashMap<(u32, u32), Vec<u64>>,
    prev_set: Vec<u32>,
    prev_prev_set: Vec<u32>,
}

impl MarkovPredictor {
    pub fn new(num_experts: usize) -> Self {
        Self {
            num_experts,
            trans1: vec![HashMap::new(); num_experts],
            total1: vec![0; num_experts],
            trans2: HashMap::new(),
            prev_set: Vec::new(),
            prev_prev_set: Vec::new(),
        }
    }

    pub fn observe_step(&mut self, prev_set: &[u32], next_set: &[u32]) {
        for &prev in prev_set {
            let row = &mut self.trans1[prev as usize];
            for &next in next_set {
                *row.entry(next).or_insert(0) += 1;
            }
            self.total1[prev as usize] += next_set.len() as u64;
        }
        for &pp in &self.prev_prev_set {
            for &p in prev_set {
                let key = (pp, p);
                let row = self.trans2.entry(key).or_insert_with(|| vec![0; self.num_experts]);
                for &n in next_set {
                    row[n as usize] += 1;
                }
            }
        }
        self.prev_prev_set = prev_set.to_vec();
        self.prev_set = next_set.to_vec();
    }

    pub fn predict_next(&self, prev_prev: Option<u32>, prev: u32) -> Vec<(u32, f64)> {
        if let Some(pp) = prev_prev {
            if let Some(row) = self.trans2.get(&(pp, prev)) {
                let total: u64 = row.iter().sum();
                if total >= MIN_2ND_ORDER_OBS {
                    return self.probabilities_from_counts(row, total);
                }
            }
        }
        let total = self.total1[prev as usize];
        if total > 0 {
            let row = &self.trans1[prev as usize];
            let counts: Vec<u64> = (0..self.num_experts)
                .map(|i| row.get(&(i as u32)).copied().unwrap_or(0))
                .collect();
            return self.probabilities_from_counts(&counts, total);
        }
        let p = 1.0 / self.num_experts as f64;
        (0..self.num_experts as u32).map(|id| (id, p)).collect()
    }

    pub fn predict_for_set(&self, _prev_prev_set: &[u32], prev_set: &[u32]) -> Vec<(u32, f64)> {
        let mut scores: Vec<(u32, f64)> = Vec::new();
        for &prev in prev_set {
            let pp = _prev_prev_set.last().copied();
            for (id, p) in self.predict_next(pp, prev) {
                match scores.iter_mut().find(|(i, _)| *i == id) {
                    Some(entry) => entry.1 = entry.1.max(p),
                    None => scores.push((id, p)),
                }
            }
        }
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }

    fn probabilities_from_counts(&self, counts: &[u64], total: u64) -> Vec<(u32, f64)> {
        let denom = total as f64 + LAPLACE_ALPHA * self.num_experts as f64;
        let mut probs: Vec<(u32, f64)> = counts
            .iter()
            .enumerate()
            .map(|(id, &c)| ((id as u32), (c as f64 + LAPLACE_ALPHA) / denom))
            .collect();
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        probs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_order_learning() {
        let mut p = MarkovPredictor::new(8);
        for _ in 0..20 {
            p.observe_step(&[0], &[1]);
        }
        let preds = p.predict_next(None, 0);
        assert_eq!(preds[0].0, 1);
        assert!(preds[0].1 > 0.7);
    }

    #[test]
    fn cold_start_is_uniform() {
        let p = MarkovPredictor::new(8);
        let preds = p.predict_next(None, 0);
        assert_eq!(preds.len(), 8);
        for (_, prob) in &preds {
            assert!((prob - 0.125).abs() < 0.001);
        }
    }
}
