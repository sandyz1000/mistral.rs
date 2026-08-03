//! Dequantization kernels for Q4_0 and Q8_0 block-quantized expert weights.
//!
//! The raw bytes on disk are in the same GGML block layout used by GGUF:
//! each block encodes `BLCK_SIZE` (32) elements and begins with a f16
//! scale `d`, followed by the quantized data.
//!
//! ## Q4_0 — 18 bytes per 32-element block
//!   `[d: f16_le (2B)] [qs: 16 × u8]` — each byte packs **two** 4-bit
//!   values (low nibble first, then high nibble). Dequant: `v = (nibble - 8) * d`.
//!
//! ## Q8_0 — 34 bytes per 32-element block
//!   `[qs: 32 × i8] [d: f16_le (2B)]` — note scale comes **after** the
//!   quantized values (different from Q4_0). Dequant: `v = qs[i] * d`.

/// Number of elements per quantized block for Q4_0 and Q8_0.
pub const BLCK_SIZE: usize = 32;

/// Super-block size for K-quants (Q4_K, Q5_K, Q8_K).
pub const QK_K: usize = 256;
/// Number of 6-bit scale bytes per super-block.
pub const K_SCALE_SIZE: usize = 12;

/// Dequantize a Q4_0 weight matrix to f32.
///
/// `data`: raw bytes from `gate_proj || up_proj || down_proj` layout.
/// `rows`: number of output rows (e.g. `d_ff`).
/// `cols`: number of output columns (e.g. `d_model`).
///
/// Returns `rows × cols` f32 values, row-major.
pub fn dequant_q4_0(data: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    let block_bytes = std::mem::size_of::<BlockQ4_0>(); // 18
    assert!(cols % BLCK_SIZE == 0, "Q4_0 cols must be multiple of {BLCK_SIZE}");
    assert_eq!(
        data.len(),
        rows * (cols / BLCK_SIZE) * block_bytes,
        "Q4_0 data size mismatch"
    );

    let blocks_per_row = cols / BLCK_SIZE;
    let mut out = vec![0.0f32; rows * cols];

    for row in 0..rows {
        let row_base = row * blocks_per_row * block_bytes;
        for blk in 0..blocks_per_row {
            let blk_offset = row_base + blk * block_bytes;
            let d = f16::from_le_bytes([data[blk_offset], data[blk_offset + 1]]).to_f32();
            for j in 0..BLCK_SIZE {
                let byte = data[blk_offset + 2 + j / 2];
                let nibble = if j % 2 == 0 {
                    byte & 0x0F
                } else {
                    byte >> 4
                };
                let col = blk * BLCK_SIZE + j;
                out[row * cols + col] = d * (nibble as f32 - 8.0);
            }
        }
    }
    out
}

/// Dequantize a Q8_0 weight matrix to f32.
///
/// `data`: raw bytes from `gate_proj || up_proj || down_proj` layout.
/// `rows`, `cols`: output dimensions (row-major).
pub fn dequant_q8_0(data: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    let block_bytes = std::mem::size_of::<BlockQ8_0>(); // 34
    assert!(cols % BLCK_SIZE == 0, "Q8_0 cols must be multiple of {BLCK_SIZE}");
    assert_eq!(
        data.len(),
        rows * (cols / BLCK_SIZE) * block_bytes,
        "Q8_0 data size mismatch"
    );

    let blocks_per_row = cols / BLCK_SIZE;
    let mut out = vec![0.0f32; rows * cols];

    for row in 0..rows {
        let row_base = row * blocks_per_row * block_bytes;
        for blk in 0..blocks_per_row {
            let blk_offset = row_base + blk * block_bytes;
            // Q8_0: 32 i8 values first, then f16 scale at offset 32
            let d = f16::from_le_bytes([data[blk_offset + 32], data[blk_offset + 33]]).to_f32();
            for j in 0..BLCK_SIZE {
                let qs = data[blk_offset + j] as i8; // byte is the quantized i8
                let col = blk * BLCK_SIZE + j;
                out[row * cols + col] = d * (qs as f32);
            }
        }
    }
    out
}

/// Decode a dtype string from the manifest (e.g. "Q4_0", "Q8_0", "Q4_K", "Q4_K_M") into
/// a dequant function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertDtype {
    F32,
    Q4_0,
    Q4_K,
    Q8_0,
}

impl ExpertDtype {
    pub fn from_manifest_str(s: &str) -> Self {
        match s.to_ascii_uppercase().as_str() {
            "Q4_0" => Self::Q4_0,
            "Q4_K" | "Q4_K_M" | "Q4_K_S" => Self::Q4_K,
            "Q8_0" => Self::Q8_0,
            _ => Self::F32,
        }
    }

    /// Expected on-disk bytes for an expert with the given dimensions.
    pub fn expert_byte_size(&self, d_ff: usize, d_model: usize) -> usize {
        let gate_up = 2 * self.proj_byte_size(d_ff, d_model);
        let down = self.proj_byte_size(d_model, d_ff);
        gate_up + down
    }

    pub fn proj_byte_size(&self, rows: usize, cols: usize) -> usize {
        match self {
            Self::F32 => rows * cols * 4,
            Self::Q4_0 => rows * (cols / BLCK_SIZE) * std::mem::size_of::<BlockQ4_0>(),
            Self::Q4_K => rows * (cols / QK_K) * std::mem::size_of::<BlockQ4K>(),
            Self::Q8_0 => rows * (cols / BLCK_SIZE) * std::mem::size_of::<BlockQ8_0>(),
        }
    }

    pub fn dequant(&self, data: &[u8], rows: usize, cols: usize) -> Vec<f32> {
        match self {
            Self::F32 => {
                assert_eq!(data.len(), rows * cols * 4);
                data.chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect()
            }
            Self::Q4_0 => dequant_q4_0(data, rows, cols),
            Self::Q4_K => dequant_q4_k(data, rows, cols),
            Self::Q8_0 => dequant_q8_0(data, rows, cols),
        }
    }
}

// These match candle-core's BlockQ4_0 / BlockQ8_0 in memory layout.
// We cannot import them directly (they're pub(crate)), so we mirror the layout.

#[repr(C)]
struct BlockQ4_0 {
    d: [u8; 2],   // f16 scale
    qs: [u8; 16], // 32 × 4-bit values
}

#[repr(C)]
struct BlockQ8_0 {
    qs: [u8; 32], // 32 × i8 values
    d: [u8; 2],   // f16 scale
}

#[repr(C)]
struct BlockQ4K {
    d: [u8; 2],           // f16 scale
    dmin: [u8; 2],        // f16 min
    scales: [u8; 12],     // 12 × 6-bit packed scales
    qs: [u8; QK_K / 2],   // 128 bytes = 256 × 4-bit nibbles
}

use half::f16;

/// Dequantize a Q4_K weight matrix to f32.
pub fn dequant_q4_k(data: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    assert!(cols % QK_K == 0, "Q4_K cols must be multiple of {QK_K}");
    let block_bytes = std::mem::size_of::<BlockQ4K>(); // 144
    let blocks_per_row = cols / QK_K;
    assert_eq!(
        data.len(),
        rows * blocks_per_row * block_bytes,
        "Q4_K data size mismatch"
    );

    let sub_blocks = QK_K / 16; // 16 sub-blocks of 16 elements each
    let mut out = vec![0.0f32; rows * cols];

    for row in 0..rows {
        let row_base = row * blocks_per_row * block_bytes;
        for blk in 0..blocks_per_row {
            let blk_offset = row_base + blk * block_bytes;
            let d = f16::from_le_bytes([data[blk_offset], data[blk_offset + 1]]).to_f32();
            let dmin = f16::from_le_bytes([data[blk_offset + 2], data[blk_offset + 3]]).to_f32();
            let scales = &data[blk_offset + 4..blk_offset + 4 + 12];
            let qs = &data[blk_offset + 4 + 12..blk_offset + block_bytes];

            for sb in 0..sub_blocks {
                let scale_byte = scales[sb / 2];
                let sc = if sb % 2 == 0 {
                    (scale_byte & 0x3F) as f32
                } else {
                    (scale_byte >> 6) as f32
                };
                let scale = d * sc + dmin;

                let sb_offset = sb * 8; // 8 bytes = 16 nibbles
                for j in 0..16 {
                    let byte = qs[sb_offset + j / 2];
                    let nibble = if j % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                    let col = blk * QK_K + sb * 16 + j;
                    out[row * cols + col] = scale * (nibble as f32 - 8.0);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_0_roundtrip() {
        // 64 elements = 2 blocks
        // Block 1: scale 2.0, values alternating 0 (low=8→0) and 1 (high=9→1)
        // Block 2: scale 1.0, values all 15 (low=15→7, high=15→7)
        let mut data = vec![0u8; 2 * std::mem::size_of::<BlockQ4_0>()];
        // Block 0: d = f16(2.0)
        let d0 = f16::from_f32(2.0);
        data[0..2].copy_from_slice(&d0.to_le_bytes());
        // qs: pairs of (0x98, 0x98...) → low nibble=8→0, high nibble=9→1
        for i in 0..16 {
            data[2 + i] = 0x98;
        }
        // Block 1: d = f16(1.0)
        let d1 = f16::from_f32(1.0);
        data[18..20].copy_from_slice(&d1.to_le_bytes());
        for i in 0..16 {
            data[20 + i] = 0xFF; // both nibbles = 15 → 7
        }

        let out = dequant_q4_0(&data, 1, 64);
        assert_eq!(out.len(), 64);
        // Block 0: 0, 1, 0, 1, ...
        for j in 0..32 {
            assert!((out[j] - if j % 2 == 0 { 0.0 } else { 2.0 }).abs() < 0.01);
        }
        // Block 1: all 7.0
        for j in 32..64 {
            assert!((out[j] - 7.0).abs() < 0.01);
        }
    }

    #[test]
    fn q8_0_roundtrip() {
        let mut data = vec![0u8; std::mem::size_of::<BlockQ8_0>()];
        // Fill qs with values -16 to 15
        for j in 0..32 {
            data[j] = (j as i8 - 16) as u8;
        }
        // scale = f16(0.5)
        let d = f16::from_f32(0.5);
        data[32..34].copy_from_slice(&d.to_le_bytes());

        let out = dequant_q8_0(&data, 1, 32);
        assert_eq!(out.len(), 32);
        for j in 0..32 {
            let expected = 0.5 * (j as f32 - 16.0);
            assert!((out[j] - expected).abs() < 0.01, "mismatch at element {j}");
        }
    }

    #[test]
    fn expert_dtype_sizes() {
        let dt32 = ExpertDtype::F32;
        let dt_q4 = ExpertDtype::Q4_0;
        let dt_q4k = ExpertDtype::Q4_K;
        let dt_q8 = ExpertDtype::Q8_0;

        // Mixtral: d_ff=14336, d_model=4096
        let f32_sz = dt32.expert_byte_size(14336, 4096);
        let q4_sz = dt_q4.expert_byte_size(14336, 4096);
        let q4k_sz = dt_q4k.expert_byte_size(14336, 4096);
        let q8_sz = dt_q8.expert_byte_size(14336, 4096);

        assert_eq!(f32_sz, (14336 * 4096 + 14336 * 4096 + 4096 * 14336) * 4);
        assert_eq!(q4_sz, (14336 * 4096 + 14336 * 4096 + 4096 * 14336) * 18 / 32);
        // Q4_K: 144 bytes per 256 elements
        assert_eq!(q4k_sz, (14336 * 4096 + 14336 * 4096 + 4096 * 14336) * 144 / 256);
        assert_eq!(q8_sz, (14336 * 4096 + 14336 * 4096 + 4096 * 14336) * 34 / 32);
    }

    #[test]
    fn q4_k_roundtrip() {
        // One super-block of 256 elements with d=2.0, dmin=0.0, uniform scales=1
        let mut data = vec![0u8; std::mem::size_of::<BlockQ4K>()];
        // d = f16(2.0)
        let d = f16::from_f32(2.0);
        data[0..2].copy_from_slice(&d.to_le_bytes());
        // dmin = f16(0.0)
        data[2..4].copy_from_slice(&[0u8, 0]);
        // scales = all 1 (packed: 0x01 in low 6 bits, 0x01<<6 in high)
        for i in 0..12 {
            data[4 + i] = 0x41; // low=1, high=1
        }
        // qs = 0x98 everywhere → low=8→0, high=9→1
        for i in 0..128 {
            data[16 + i] = 0x98;
        }

        let out = dequant_q4_k(&data, 1, 256);
        assert_eq!(out.len(), 256);
        // Scale = d * 1 + 0 = 2.0. Even indices = 0, odd = 2.0 * 1 = 2.0
        for j in 0..256 {
            let expected = if j % 2 == 0 { 0.0 } else { 2.0 };
            assert!((out[j] - expected).abs() < 0.01, "mismatch at {j}: got {}", out[j]);
        }
    }
}
