//! Extract MoE expert tensors from a GGUF file into per-expert binary files.
//!
//! Reads a GGUF file, identifies expert tensors by naming pattern, writes:
//! - `dense.gguf` — symlink to original (Phase 1.1 will build a proper subset)
//! - `experts/layer_{L}_expert_{E}.bin` — gate_proj || up_proj || down_proj
//! - `manifest.json` — expert metadata (counts, shapes, dtypes, file sizes)

use candle_core::quantized::{gguf_file, GgmlDType};
use clap::Parser;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Seek, Write};
use std::path::PathBuf;

// ── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "mistralrs-expert-extract")]
#[command(about = "Split a GGUF MoE model into dense tensors + per-expert files")]
struct Args {
    #[arg(short, long)]
    input: PathBuf,
    #[arg(short, long)]
    output_dir: PathBuf,
}

// ── Manifest types ───────────────────────────────────────────────────────────

#[derive(Serialize)]
struct Manifest {
    num_layers: usize,
    num_experts_per_layer: usize,
    layers: Vec<LayerInfo>,
    expert_map: BTreeMap<u32, ExpertEntry>,
}

#[derive(Serialize)]
struct LayerInfo {
    layer_idx: usize,
    num_experts: usize,
    expert_ids: Vec<u32>,
}

#[derive(Serialize)]
struct ExpertEntry {
    layer_idx: usize,
    local_id: usize,
    file: String,
    dtype: String,
    d_model: usize,
    d_ff: usize,
    byte_size: u64,
}

// ── Internal ─────────────────────────────────────────────────────────────────

struct ProjMeta {
    shape: Vec<usize>,
    ggml_dtype: GgmlDType,
    offset: u64,
}

type ThreeProjs = [Option<ProjMeta>; 3];

// ── Name parsing ─────────────────────────────────────────────────────────────

fn parse_expert_tensor(name: &str) -> Option<(usize, usize, &str)> {
    // Pattern 1: blk.{L}.ffn_gate.{E}.weight / ffn_up / ffn_down (per-expert)
    if let Some(rest) = name.strip_prefix("blk.") {
        let (layer_str, rest) = rest.split_once('.')?;
        let layer: usize = layer_str.parse().ok()?;
        for (prefix, proj) in [("ffn_gate.", "gate"), ("ffn_up.", "up"), ("ffn_down.", "down")] {
            if let Some(rest) = rest.strip_prefix(prefix) {
                let expert: usize = rest.strip_suffix(".weight")?.parse().ok()?;
                return Some((layer, expert, proj));
            }
        }
        // Pattern 1b: blk.{L}.ffn_gate_exps.weight (fused, Qwen3/DeepSeek style)
        // We return expert=0 for all; caller splits by stride.
        for (suffix, proj) in [("ffn_gate_exps.weight", "gate"), ("ffn_up_exps.weight", "up"), ("ffn_down_exps.weight", "down")] {
            if rest == suffix {
                return Some((layer, 0, proj));
            }
        }
        return None;
    }
    // Pattern 2: model.layers.{L}.block_sparse_moe.experts.{E}.{proj}.weight
    if let Some(rest) = name.strip_prefix("model.layers.") {
        let parts: Vec<&str> = rest.split('.').collect();
        if parts.len() < 4 {
            return None;
        }
        let layer: usize = parts[0].parse().ok()?;
        let exp_pos = parts.iter().position(|&p| p == "experts")?;
        if exp_pos + 2 >= parts.len() {
            return None;
        }
        let expert: usize = parts[exp_pos + 1].parse().ok()?;
        let proj = classify_proj(parts[exp_pos + 2])?;
        return Some((layer, expert, proj));
    }
    // Pattern 3: transformer.h.{L}.moe.experts.{E}.{proj}.weight
    if let Some(rest) = name.strip_prefix("transformer.h.") {
        let (layer_str, rest) = rest.split_once('.')?;
        let layer: usize = layer_str.parse().ok()?;
        let after = &rest[rest.find(".experts.")? + ".experts.".len()..];
        let (expert_str, proj_name) = after.split_once('.')?;
        let expert: usize = expert_str.parse().ok()?;
        let proj = classify_proj(proj_name)?;
        return Some((layer, expert, proj));
    }
    None
}

fn classify_proj(name: &str) -> Option<&'static str> {
    match name {
        "w1" | "gate_proj" => Some("gate"),
        "w3" | "up_proj" => Some("up"),
        "w2" | "down_proj" => Some("down"),
        _ => None,
    }
}

fn is_router_gate(name: &str) -> bool {
    name.contains("ffn_gate_inp")
        || name.ends_with(".block_sparse_moe.gate.weight")
        || (name.contains(".gate.") && !name.contains("experts"))
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let input_file = File::open(&args.input)?;
    let mut reader = BufReader::new(input_file);
    let ct = gguf_file::Content::read(&mut reader)
        .map_err(|e| anyhow::anyhow!("Failed to read GGUF: {e}"))?;

    let arch = ct
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok().cloned())
        .unwrap_or_else(|| "unknown".into());
    println!("Architecture: {arch}");
    println!("Total tensors: {}", ct.tensor_infos.len());

    // ── Classify ────────────────────────────────────────────────────────
    let mut expert_groups: BTreeMap<(usize, usize), ThreeProjs> = BTreeMap::new();
    let mut dense_tensors: Vec<String> = Vec::new();
    let mut max_layer = 0usize;

    for (name, info) in &ct.tensor_infos {
        if let Some((layer, expert, proj)) = parse_expert_tensor(name) {
            max_layer = max_layer.max(layer);
            let entry = expert_groups.entry((layer, expert)).or_default();
            let idx = match proj {
                "gate" => 0,
                "up" => 1,
                "down" => 2,
                _ => continue,
            };
            entry[idx] = Some(ProjMeta {
                shape: info.shape.dims().to_vec(),
                ggml_dtype: info.ggml_dtype,
                offset: info.offset,
            });
        } else if is_router_gate(name) {
            dense_tensors.push(name.clone());
        } else {
            dense_tensors.push(name.clone());
        }
    }

    // ── Detect fused expert tensors and expand ───────────────────────
    // Qwen3/DeepSeek store experts fused: blk.L.ffn_gate_exps.weight has shape
    // [num_experts, d_ff, d_model]. Split into per-expert projections.
    {
        let mut fused: BTreeMap<(usize, usize), ThreeProjs> = BTreeMap::new();
        let keys: Vec<(usize, usize)> = expert_groups.keys().cloned().collect();
        for (layer, expert) in keys {
            let projs = &expert_groups[&(layer, expert)];
            // Check if any projection is fused (shape.len() == 3)
            let gate_3d = projs[0].as_ref().map(|m| m.shape.len() == 3).unwrap_or(false);
            let up_3d = projs[1].as_ref().map(|m| m.shape.len() == 3).unwrap_or(false);
            let down_3d = projs[2].as_ref().map(|m| m.shape.len() == 3).unwrap_or(false);
            if !gate_3d && !up_3d && !down_3d {
                continue;
            }
            // Extract per-expert shapes
            let num_e = if gate_3d {
                projs[0].as_ref().unwrap().shape[0]
            } else if up_3d {
                projs[1].as_ref().unwrap().shape[0]
            } else {
                projs[2].as_ref().unwrap().shape[0]
            };
            for e_idx in 0..num_e {
                let mut entry = [None, None, None];
                for (p_idx, is_3d) in [(0, gate_3d), (1, up_3d), (2, down_3d)] {
                    if let Some(ref m) = projs[p_idx] {
                        let byte_stride: u64 = if is_3d {
                            let elem_count: usize = m.shape[1..].iter().product();
                            (elem_count / m.ggml_dtype.block_size().max(1) as usize
                                * m.ggml_dtype.type_size() as usize) as u64
                        } else {
                            0
                        };
                        entry[p_idx] = Some(ProjMeta {
                            shape: if is_3d { m.shape[1..].to_vec() } else { m.shape.clone() },
                            ggml_dtype: m.ggml_dtype,
                            offset: if is_3d {
                                m.offset + e_idx as u64 * byte_stride
                            } else {
                                m.offset
                            },
                        });
                    }
                }
                fused.insert((layer, e_idx), entry);
            }
            max_layer = max_layer.max(layer);
        }
        expert_groups = fused;
    }

    let num_layers = max_layer + 1;
    let (first_layer, first_expert) = expert_groups
        .keys()
        .next()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("No expert tensors found. Not an MoE model?"))?;
    let num_experts_per_layer = expert_groups
        .keys()
        .filter(|(l, _)| *l == first_layer)
        .count();
    println!(
        "MoE: {num_layers} layers × {num_experts_per_layer} experts/layer = {} experts",
        expert_groups.len()
    );
    println!("Dense tensors to keep: {}", dense_tensors.len());

    let gate_meta = expert_groups[&(first_layer, first_expert)][0]
        .as_ref()
        .expect("missing gate projection");
    let d_model = gate_meta.shape.get(1).copied().unwrap_or(0);
    let d_ff = gate_meta.shape[0];
    let dtype_str = format!("{:?}", gate_meta.ggml_dtype);
    println!("d_model={d_model}, d_ff={d_ff}, dtype={dtype_str}");

    // ── Write outputs ───────────────────────────────────────────────────
    let experts_dir = args.output_dir.join("experts");
    fs::create_dir_all(&experts_dir)?;

    let mut manifest = Manifest {
        num_layers,
        num_experts_per_layer,
        layers: Vec::new(),
        expert_map: BTreeMap::new(),
    };

    for layer_idx in 0..num_layers {
        let mut expert_ids = Vec::new();
        for expert_idx in 0..num_experts_per_layer {
            let Some(projs) = expert_groups.get(&(layer_idx, expert_idx)) else {
                eprintln!("WARNING: missing L{layer_idx}_E{expert_idx}, skipping");
                continue;
            };
            let (Some(gate), Some(up), Some(down)) = (&projs[0], &projs[1], &projs[2]) else {
                eprintln!("WARNING: incomplete projections for L{layer_idx}_E{expert_idx}");
                continue;
            };

            let file_name = format!("layer_{layer_idx}_expert_{expert_idx}.bin");
            let out_path = experts_dir.join(&file_name);
            let mut out = BufWriter::new(File::create(&out_path)?);
            let mut total: u64 = 0;

            for meta in [gate, up, down] {
                let raw = read_raw_tensor(
                    meta.ggml_dtype,
                    meta.shape.iter().product::<usize>(),
                    meta.offset,
                    &mut reader,
                    ct.tensor_data_offset,
                )?;
                out.write_all(&raw)?;
                total += raw.len() as u64;
            }
            out.flush()?;

            let global_id = (layer_idx * num_experts_per_layer + expert_idx) as u32;
            expert_ids.push(global_id);
            manifest.expert_map.insert(
                global_id,
                ExpertEntry {
                    layer_idx,
                    local_id: expert_idx,
                    file: format!("experts/{file_name}"),
                    dtype: dtype_str.clone(),
                    d_model,
                    d_ff,
                    byte_size: total,
                },
            );
        }
        manifest.layers.push(LayerInfo {
            layer_idx,
            num_experts: expert_ids.len(),
            expert_ids,
        });
    }

    fs::write(
        args.output_dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;

    // dense.gguf symlink
    let link_path = args.output_dir.join("dense.gguf");
    let target = args.input.canonicalize().unwrap_or_else(|_| args.input.clone());
    if link_path.exists() {
        fs::remove_file(&link_path)?;
    }
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(&target, &link_path)?;
    }
    #[cfg(not(unix))]
    {
        fs::copy(&target, &link_path)?;
    }

    println!("Done. {} → {}", manifest.expert_map.len(), args.output_dir.display());
    Ok(())
}

// ── Raw tensor reader ────────────────────────────────────────────────────────

fn read_raw_tensor(
    ggml_dtype: GgmlDType,
    elem_count: usize,
    offset: u64,
    reader: &mut BufReader<File>,
    tensor_data_offset: u64,
) -> anyhow::Result<Vec<u8>> {
    let block = ggml_dtype.block_size().max(1) as usize;
    let type_sz = ggml_dtype.type_size() as usize;
    let byte_len = elem_count / block * type_sz;
    let mut buf = vec![0u8; byte_len];
    reader.seek(std::io::SeekFrom::Start(tensor_data_offset + offset))?;
    reader.read_exact(&mut buf)?;
    Ok(buf)
}
