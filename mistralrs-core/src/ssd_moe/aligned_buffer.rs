//! Page-aligned heap buffers for `O_DIRECT`.
//!
//! `Vec<u8>` only guarantees `align_of::<u8>() == 1`. The kernel rejects
//! `O_DIRECT` reads whose buffer address and length are not aligned to the
//! device's logical block size (typically 4096). This module provides
//! [`AlignedBuffer`], a wrapper around `std::alloc::alloc` with a chosen
//! alignment.

use std::alloc::{self, Layout};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

/// A heap-allocated, page-aligned, zero-initialised buffer for `O_DIRECT` I/O.
pub struct AlignedBuffer {
    ptr: NonNull<u8>,
    len: usize,
    align: usize,
}

unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

impl AlignedBuffer {
    pub fn new(size: usize, align: usize) -> Self {
        assert!(align.is_power_of_two(), "alignment must be a power of two");
        assert!(size > 0, "buffer size must be > 0");
        assert!(
            size % align == 0,
            "size {size} must be a multiple of alignment {align} for O_DIRECT"
        );

        let layout = Layout::from_size_align(size, align).expect("invalid layout");
        let raw = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(raw).unwrap_or_else(|| alloc::handle_alloc_error(layout));
        Self { ptr, len: size, align }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn align(&self) -> usize {
        self.align
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.len, self.align).expect("invalid layout");
        unsafe { alloc::dealloc(self.ptr.as_ptr(), layout) };
    }
}

impl Deref for AlignedBuffer {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl DerefMut for AlignedBuffer {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

impl AsRef<[u8]> for AlignedBuffer {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsMut<[u8]> for AlignedBuffer {
    fn as_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_is_aligned_and_zeroed() {
        let buf = AlignedBuffer::new(4096, 4096);
        assert_eq!(buf.as_slice().as_ptr() as usize % 4096, 0);
        assert_eq!(buf.len(), 4096);
        assert!(buf.iter().all(|&b| b == 0));
    }

    #[test]
    #[should_panic]
    fn rejects_non_power_of_two_alignment() {
        AlignedBuffer::new(1024, 1000);
    }

    #[test]
    #[should_panic]
    fn rejects_size_not_multiple_of_alignment() {
        AlignedBuffer::new(4097, 4096);
    }
}
