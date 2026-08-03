//! In-RAM LRU cache of resident experts.
//!
//! Each entry wraps a `PooledBuffer` held behind an `Arc`. Eviction drops the
//! `Arc`; once in-flight inference also drops its handle, the buffer returns
//! to the pool's free list automatically.

use super::buffer_pool::PooledBuffer;
use lru::LruCache;
use parking_lot::Mutex;
use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::sync::Arc;

/// One resident expert: id + raw bytes from SSD.
pub struct ExpertResident {
    pub id: u32,
    pub buffer: PooledBuffer,
}

impl ExpertResident {
    pub fn new(id: u32, buffer: PooledBuffer) -> Self {
        Self { id, buffer }
    }

    #[inline]
    pub fn data(&self) -> &[u8] {
        self.buffer.as_slice()
    }
}

/// Thread-safe fixed-capacity LRU cache of resident experts.
pub struct ExpertCache {
    inner: Mutex<LruCache<u32, Arc<ExpertResident>>>,
    pinned: Mutex<HashSet<u32>>,
    capacity: usize,
}

impl ExpertCache {
    pub fn new(capacity: usize) -> Self {
        let cap =
            NonZeroUsize::new(capacity).expect("cache capacity must be > 0");
        Self {
            inner: Mutex::new(LruCache::new(cap)),
            pinned: Mutex::new(HashSet::new()),
            capacity,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Look up an expert. Updates LRU recency on hit.
    pub fn get(&self, id: u32) -> Option<Arc<ExpertResident>> {
        self.inner.lock().get(&id).cloned()
    }

    /// Check residency without changing recency.
    pub fn contains(&self, id: u32) -> bool {
        self.inner.lock().peek(&id).is_some()
    }

    /// Insert a resident. Returns the evicted resident if any, or `Err` if
    /// the cache is full and every resident is pinned.
    pub fn insert(
        &self,
        resident: Arc<ExpertResident>,
    ) -> Result<Option<Arc<ExpertResident>>, Arc<ExpertResident>> {
        let id = resident.id;
        let pinned = self.pinned.lock();
        let mut guard = self.inner.lock();

        let mut pre_evicted = None;
        if guard.len() >= self.capacity && guard.peek(&id).is_none() {
            let victim = self.select_victim_id(&guard, &pinned);
            match victim {
                Some(v) => pre_evicted = guard.pop(&v),
                None => return Err(resident),
            }
        }
        let push_evicted = guard.push(id, resident).map(|(_, v)| v);
        Ok(push_evicted.or(pre_evicted))
    }

    /// Choose the least-recently-used non-pinned id.
    fn select_victim_id(
        &self,
        guard: &LruCache<u32, Arc<ExpertResident>>,
        pinned: &HashSet<u32>,
    ) -> Option<u32> {
        // `iter()` yields MRU-first, so the last non-pinned id is LRU.
        guard
            .iter()
            .map(|(k, _)| *k)
            .filter(|k| !pinned.contains(k))
            .last()
    }

    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    /// Pop the least-recently-used non-pinned entry.
    pub fn evict_lru(&self) -> Option<Arc<ExpertResident>> {
        let pinned = self.pinned.lock();
        if pinned.is_empty() {
            return self.inner.lock().pop_lru().map(|(_, v)| v);
        }
        let mut guard = self.inner.lock();
        let victim = self.select_victim_id(&guard, &pinned)?;
        guard.pop(&victim)
    }

    pub fn pin(&self, id: u32) {
        self.pinned.lock().insert(id);
    }

    pub fn unpin(&self, id: u32) {
        self.pinned.lock().remove(&id);
    }

    pub fn is_pinned(&self, id: u32) -> bool {
        self.pinned.lock().contains(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssd_moe::buffer_pool::BufferPool;

    fn make(id: u32, pool: &BufferPool) -> Arc<ExpertResident> {
        let buf = pool.try_acquire().unwrap();
        Arc::new(ExpertResident::new(id, buf))
    }

    #[test]
    fn lru_eviction_returns_buffer_to_pool() {
        let pool = BufferPool::new(3, 4096, 4096);
        let cache = ExpertCache::new(2);

        cache.insert(make(0, &pool)).ok().unwrap();
        cache.insert(make(1, &pool)).ok().unwrap();
        let scratch = pool.try_acquire().unwrap();
        assert!(pool.try_acquire().is_none());
        drop(scratch);

        let evicted = match cache.insert(make(2, &pool)) {
            Ok(Some(e)) => e,
            other => panic!("expected Ok(Some(_)), got {:?}", other.is_ok()),
        };
        assert_eq!(evicted.id, 0);
        drop(evicted);
        assert!(pool.try_acquire().is_some());
    }

    #[test]
    fn hit_updates_recency() {
        let pool = BufferPool::new(3, 4096, 4096);
        let cache = ExpertCache::new(2);
        cache.insert(make(0, &pool)).ok().unwrap();
        cache.insert(make(1, &pool)).ok().unwrap();
        let _ = cache.get(0);
        cache.insert(make(2, &pool)).ok().unwrap();
        assert!(cache.contains(0));
        assert!(!cache.contains(1));
    }

    #[test]
    fn pinned_protected_from_eviction() {
        let pool = BufferPool::new(4, 4096, 4096);
        let cache = ExpertCache::new(2);
        cache.insert(make(0, &pool)).ok().unwrap();
        cache.insert(make(1, &pool)).ok().unwrap();
        cache.pin(0);
        let evicted = match cache.insert(make(2, &pool)) {
            Ok(Some(e)) => e,
            other => panic!("expected Ok(Some(_))"),
        };
        assert_eq!(evicted.id, 1);
        assert!(cache.contains(0));
    }
}
