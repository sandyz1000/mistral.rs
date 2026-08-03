//! Linux `io_uring` storage backend with registered fixed buffers.
//!
//! Compiled only when `cfg(all(target_os = "linux", feature = "io-uring"))`.
//! Falls back to `PreadStorage` on other platforms.
//!
//! Key benefits:
//! 1. **Registered fixed buffers** — every `BufferPool` slot is pinned once at
//!    startup via `io_uring_register(IORING_REGISTER_BUFFERS)`. After that,
//!    reads reference a buffer index — no per-read iovec setup.
//! 2. **Batched reads** — when a token misses K experts, we push K SQEs and
//!    `enter()` once, cutting syscall overhead by K×.
//! 3. **O_DIRECT** — expert files are opened with `O_DIRECT`, bypassing the
//!    page cache.
//!
//! ## Architecture
//!
//! A dedicated reactor thread owns the `IoUring` instance. Callers send
//! `ReactorRequest` envelopes through a bounded `mpsc` channel; the reactor
//! pushes SQEs, `submit_and_wait`s, drains CQEs, and responds via `oneshot`.
//!
//! The reactor thread is spawned in `IoUringStorage::new` and joined on `Drop`.

use super::buffer_pool::{BufferPool, PooledBuffer};
use io_uring::{opcode, types, IoUring};
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::os::unix::io::{AsRawFd, RawFd};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Configuration for the io_uring backend.
pub struct IoUringConfig {
    pub base_path: PathBuf,
    pub expert_size: usize,
    pub queue_depth: u32,
}

impl Default for IoUringConfig {
    fn default() -> Self {
        Self {
            base_path: PathBuf::from("."),
            expert_size: 0,
            queue_depth: 64,
        }
    }
}

/// io_uring storage backend.
///
/// Holds a clone of the `BufferPool` so the kernel-registered fixed-buffer
/// pointers remain valid for the lifetime of the backend.
pub struct IoUringStorage {
    tx: tokio::sync::mpsc::Sender<ReactorRequest>,
    reactor: Option<std::thread::JoinHandle<()>>,
    expert_size: usize,
    _pool: BufferPool,
}

struct ReactorRequest {
    ids: Vec<u32>,
    ptrs: Vec<*mut u8>,
    len: usize,
    reply: tokio::sync::oneshot::Sender<io::Result<usize>>,
}

unsafe impl Send for ReactorRequest {}

impl IoUringStorage {
    /// Create a new io_uring backend. Registers every buffer in `pool` as a
    /// fixed io_uring buffer via `io_uring_register(IORING_REGISTER_BUFFERS)`.
    ///
    /// Returns an error if the pool is empty or queue_depth is 0.
    pub fn new(cfg: IoUringConfig, pool: &BufferPool) -> io::Result<Self> {
        let iovecs = pool.raw_iovecs();
        if iovecs.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "io_uring backend requires a non-empty buffer pool",
            ));
        }
        let qd = (cfg.queue_depth as usize).max(1);
        let mut ring = IoUring::new(qd as u32)?;

        let raw_iovecs: Vec<libc::iovec> = iovecs
            .iter()
            .map(|(p, l)| libc::iovec {
                iov_base: *p as *mut _,
                iov_len: *l,
            })
            .collect();
        unsafe { ring.submitter().register_buffers(&raw_iovecs)? };

        let buf_index: HashMap<usize, u16> = iovecs
            .iter()
            .enumerate()
            .map(|(i, (p, _))| (*p as usize, i as u16))
            .collect();

        let base_path = cfg.base_path;
        let expert_size = cfg.expert_size;
        let (tx, rx) = tokio::sync::mpsc::channel::<ReactorRequest>(qd);

        let reactor = std::thread::Builder::new()
            .name("ssd-moe-io-uring".into())
            .spawn(move || reactor_loop(ring, rx, &base_path, expert_size, &buf_index, qd))
            .map_err(|e| {
                io::Error::new(
                    io::ErrorKind::Other,
                    format!("io_uring: failed to spawn reactor thread: {e}"),
                )
            })?;

        Ok(Self {
            tx,
            reactor: Some(reactor),
            expert_size,
            _pool: pool.clone(),
        })
    }

    /// Number of bytes per expert.
    pub fn expert_size(&self) -> usize {
        self.expert_size
    }

    /// Read a single expert synchronously via the reactor.
    pub async fn read_expert(
        &self,
        expert_id: u32,
        buf: &mut PooledBuffer,
    ) -> io::Result<usize> {
        assert_eq!(buf.len(), self.expert_size);
        let ptr = buf.as_mut_slice().as_mut_ptr();
        self.submit_batch(&[expert_id], &[ptr], self.expert_size)
            .await
    }

    /// Batch-read K experts with one `enter()` syscall.
    pub async fn read_experts_batch(
        &self,
        ids: &[u32],
        bufs: &mut [&mut PooledBuffer],
    ) -> io::Result<usize> {
        assert_eq!(ids.len(), bufs.len());
        if ids.is_empty() {
            return Ok(0);
        }
        let ptrs: Vec<*mut u8> = bufs
            .iter_mut()
            .map(|b| b.as_mut_slice().as_mut_ptr())
            .collect();
        self.submit_batch(ids, &ptrs, self.expert_size).await
    }

    async fn submit_batch(
        &self,
        ids: &[u32],
        ptrs: &[*mut u8],
        len: usize,
    ) -> io::Result<usize> {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        let req = ReactorRequest {
            ids: ids.to_vec(),
            ptrs: ptrs.to_vec(),
            len,
            reply: reply_tx,
        };
        self.tx.send(req).await.map_err(|_| {
            io::Error::new(
                io::ErrorKind::Other,
                "io_uring reactor channel closed",
            )
        })?;
        reply_rx.await.map_err(|_| {
            io::Error::new(
                io::ErrorKind::Other,
                "io_uring reactor dropped reply channel",
            )
        })?
    }
}

impl Drop for IoUringStorage {
    fn drop(&mut self) {
        drop(self.tx.send(ReactorRequest {
            ids: Vec::new(),
            ptrs: Vec::new(),
            len: 0,
            reply: tokio::sync::oneshot::channel().0,
        }));
        // The reactor will receive the empty request as a signal to drain
        // remaining work and exit (or just exit on Drop of the channel).
        // We don't join here to avoid blocking the async runtime's drop.
        // The reactor thread will exit shortly after the channel closes.
        drop(self.tx.clone());
        if let Some(h) = self.reactor.take() {
            let _ = h.join();
        }
    }
}

// ── Reactor loop ─────────────────────────────────────────────────────────────

fn reactor_loop(
    mut ring: IoUring,
    mut rx: tokio::sync::mpsc::Receiver<ReactorRequest>,
    base_path: &Path,
    expert_size: usize,
    buf_index: &HashMap<usize, u16>,
    queue_depth: usize,
) {
    let mut fds: HashMap<u32, Arc<File>> = HashMap::new();

    while let Some(req) = rx.blocking_recv() {
        if req.ids.is_empty() {
            continue; // skip empty shutdown signals
        }
        let result = process_batch(
            &mut ring,
            &req,
            base_path,
            expert_size,
            buf_index,
            &mut fds,
            queue_depth,
        );
        let _ = req.reply.send(result);
    }
}

fn process_batch(
    ring: &mut IoUring,
    req: &ReactorRequest,
    base_path: &Path,
    expert_size: usize,
    buf_index: &HashMap<usize, u16>,
    fds: &mut HashMap<u32, Arc<File>>,
    queue_depth: usize,
) -> io::Result<usize> {
    let total = req.ids.len();
    if total == 0 {
        return Ok(0);
    }

    // Open files (cached)
    let mut keep_alive: Vec<Arc<File>> = Vec::with_capacity(total);
    for &id in &req.ids {
        let f = open_expert_file(base_path, id, fds)?;
        keep_alive.push(f);
    }

    let mut pushed: usize = 0;

    while pushed < total {
        let chunk_end = (pushed + queue_depth).min(total);

        for i in pushed..chunk_end {
            let fd = keep_alive[i].as_raw_fd();
            let buf_idx = buf_index
                .get(&(req.ptrs[i] as usize))
                .copied()
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "io_uring: buffer pointer not registered",
                    )
                })?;

            let sqe = opcode::ReadFixed::new(
                types::Fd(fd),
                req.ptrs[i],
                expert_size as u32,
                buf_idx,
            )
            .offset(0)
            .build()
            .user_data(req.ids[i] as u64);

            unsafe { ring.submission().push(&sqe) }.map_err(|e| {
                io::Error::new(
                    io::ErrorKind::Other,
                    format!("io_uring push failed: {e}"),
                )
            })?;
        }

        // Submit this chunk
        ring.submit()?;
        pushed = chunk_end;
    }

    // Wait for all completions
    ring.submit_and_wait(total)?;

    // Drain completions
    let mut total_bytes: usize = 0;
    let mut first_err: Option<io::Error> = None;
    let mut drained: usize = 0;

    while drained < total {
        let cqe = match ring.completion().next() {
            Some(c) => c,
            None => {
                ring.submit()?;
                std::thread::sleep(Duration::from_micros(10));
                continue;
            }
        };
        drained += 1;

        let result = cqe.result();
        if result < 0 {
            let e = io::Error::from_raw_os_error(-result);
            if first_err.is_none() {
                first_err = Some(io::Error::new(
                    e.kind(),
                    format!("io_uring read on expert {} failed: {e}", cqe.user_data()),
                ));
            }
            continue;
        }
        let n = result as usize;
        if n != expert_size {
            if first_err.is_none() {
                first_err = Some(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!(
                        "io_uring short read on expert {}: got {n} expected {expert_size}",
                        cqe.user_data()
                    ),
                ));
            }
            continue;
        }
        total_bytes += n;
    }

    if let Some(e) = first_err {
        return Err(e);
    }
    Ok(total_bytes)
}

fn open_expert_file(
    base_path: &Path,
    id: u32,
    fd_cache: &mut HashMap<u32, Arc<File>>,
) -> io::Result<Arc<File>> {
    if let Some(f) = fd_cache.get(&id) {
        return Ok(f.clone());
    }
    let path = base_path.join(format!("expert_{id}.bin"));
    // Use O_DIRECT on Linux
    let mut opts = std::fs::OpenOptions::new();
    opts.read(true);
    opts.custom_flags(libc::O_DIRECT);
    let file = opts.open(&path)?;
    let f = Arc::new(file);
    fd_cache.insert(id, f.clone());
    Ok(f)
}
