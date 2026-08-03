//! Fixed-size slab pool of pre-allocated, page-aligned RAM buffers.
//!
//! At startup we allocate `slots` aligned buffers of `expert_size` bytes and
//! hand them out as [`PooledBuffer`] RAII guards. When a guard is dropped its
//! buffer is returned to the pool's free list — LRU eviction of an expert
//! automatically frees a slot for the next miss.
//!
//! ## Primary / Shadow split
//!
//! When speculative prefetch is enabled, the engine can overlap the current
//! token's compute with the *next* token's prefetch. The **shadow** half
//! provides independent buffers so speculation never starves real work on the
//! primary pool. Buffers can be promoted from shadow to primary via
//! [`BufferPool::promote_shadow`] when a speculative fetch is confirmed.

use super::aligned_buffer::AlignedBuffer;
use parking_lot::Mutex;
use std::sync::Arc;
use tokio::sync::Notify;

struct Inner {
    free: Mutex<Vec<AlignedBuffer>>,
    shadow: Option<Mutex<Vec<AlignedBuffer>>>,
    notify: Notify,
    notify_shadow: Notify,
    primary_slots: usize,
    shadow_slots: usize,
    buffer_size: usize,
    align: usize,
}

#[derive(Clone)]
pub struct BufferPool {
    inner: Arc<Inner>,
}

impl BufferPool {
    pub fn new(slots: usize, buffer_size: usize, align: usize) -> Self {
        Self::new_with_shadow(slots, 0, buffer_size, align)
    }

    pub fn new_with_shadow(
        primary_slots: usize,
        shadow_slots: usize,
        buffer_size: usize,
        align: usize,
    ) -> Self {
        assert!(primary_slots > 0, "primary pool must have at least one slot");
        let mut free = Vec::with_capacity(primary_slots);
        for _ in 0..primary_slots {
            free.push(AlignedBuffer::new(buffer_size, align));
        }
        let shadow = if shadow_slots > 0 {
            let mut s = Vec::with_capacity(shadow_slots);
            for _ in 0..shadow_slots {
                s.push(AlignedBuffer::new(buffer_size, align));
            }
            Some(Mutex::new(s))
        } else {
            None
        };
        Self {
            inner: Arc::new(Inner {
                free: Mutex::new(free),
                shadow,
                notify: Notify::new(),
                notify_shadow: Notify::new(),
                primary_slots,
                shadow_slots,
                buffer_size,
                align,
            }),
        }
    }

    pub fn capacity(&self) -> usize {
        self.inner.primary_slots
    }

    pub fn shadow_capacity(&self) -> usize {
        self.inner.shadow_slots
    }

    pub fn buffer_size(&self) -> usize {
        self.inner.buffer_size
    }

    pub fn try_acquire(&self) -> Option<PooledBuffer> {
        let buf = self.inner.free.lock().pop()?;
        Some(PooledBuffer {
            buffer: Some(buf),
            pool: self.inner.clone(),
            is_shadow: false,
        })
    }

    pub fn try_acquire_shadow(&self) -> Option<PooledBuffer> {
        let shadow = self.inner.shadow.as_ref()?;
        let buf = shadow.lock().pop()?;
        Some(PooledBuffer {
            buffer: Some(buf),
            pool: self.inner.clone(),
            is_shadow: true,
        })
    }

    pub fn promote_shadow(&self, mut buf: PooledBuffer) -> PooledBuffer {
        if buf.is_shadow {
            buf.is_shadow = false;
        }
        buf
    }

    /// Pop a free **primary** buffer, waiting asynchronously if none are free.
    pub async fn acquire(&self) -> PooledBuffer {
        loop {
            if let Some(b) = self.try_acquire() {
                return b;
            }
            let notified = self.inner.notify.notified();
            if let Some(b) = self.try_acquire() {
                return b;
            }
            notified.await;
        }
    }

    /// Snapshot `(ptr, len)` of every currently-free buffer for io_uring
    /// fixed-buffer registration.
    pub fn raw_iovecs(&self) -> Vec<(*mut u8, usize)> {
        let mut out =
            Vec::with_capacity(self.inner.primary_slots + self.inner.shadow_slots);
        {
            let mut g = self.inner.free.lock();
            out.extend(g.iter_mut().map(|b| (b.as_mut_slice().as_mut_ptr(), b.len())));
        }
        if let Some(s) = &self.inner.shadow {
            let mut g = s.lock();
            out.extend(g.iter_mut().map(|b| (b.as_mut_slice().as_mut_ptr(), b.len())));
        }
        out
    }
}

/// RAII guard wrapping an `AlignedBuffer` borrowed from a `BufferPool`.
pub struct PooledBuffer {
    buffer: Option<AlignedBuffer>,
    pool: Arc<Inner>,
    is_shadow: bool,
}

impl PooledBuffer {
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        self.buffer.as_ref().expect("PooledBuffer must hold a buffer until Drop").as_slice()
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.buffer.as_mut().expect("PooledBuffer must hold a buffer until Drop").as_mut_slice()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.buffer.as_ref().expect("PooledBuffer must hold a buffer until Drop").len()
    }

    #[inline]
    pub fn is_shadow(&self) -> bool {
        self.is_shadow
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if let Some(buf) = self.buffer.take() {
            if self.is_shadow {
                if let Some(s) = &self.pool.shadow {
                    s.lock().push(buf);
                    self.pool.notify_shadow.notify_one();
                } else {
                    self.pool.free.lock().push(buf);
                    self.pool.notify.notify_one();
                }
            } else {
                self.pool.free.lock().push(buf);
                self.pool.notify.notify_one();
            }
        }
    }
}

impl AsRef<[u8]> for PooledBuffer {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsMut<[u8]> for PooledBuffer {
    fn as_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn acquire_and_release() {
        let pool = BufferPool::new(2, 4096, 4096);
        let a = pool.acquire().await;
        let b = pool.acquire().await;
        assert!(pool.try_acquire().is_none());
        drop(a);
        assert!(pool.try_acquire().is_some());
        drop(b);
    }

    #[test]
    fn shadow_independent_of_primary() {
        let pool = BufferPool::new_with_shadow(2, 2, 4096, 4096);
        assert_eq!(pool.capacity(), 2);
        assert_eq!(pool.shadow_capacity(), 2);

        let _a = pool.try_acquire().unwrap();
        let _b = pool.try_acquire().unwrap();
        assert!(pool.try_acquire().is_none());

        let _s1 = pool.try_acquire_shadow().unwrap();
        let _s2 = pool.try_acquire_shadow().unwrap();
        assert!(pool.try_acquire_shadow().is_none());
    }

    #[test]
    fn promote_shadow_to_primary() {
        let pool = BufferPool::new_with_shadow(1, 1, 4096, 4096);
        let _hold = pool.try_acquire().unwrap();
        let s = pool.try_acquire_shadow().unwrap();
        assert!(s.is_shadow());
        let promoted = pool.promote_shadow(s);
        assert!(!promoted.is_shadow());
        drop(promoted);
        let _p = pool.try_acquire().expect("promoted buffer returned to primary");
        assert!(pool.try_acquire_shadow().is_none());
    }
}
