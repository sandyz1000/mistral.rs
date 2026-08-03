#[cfg(test)]
mod tests {
    use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
    use std::fs;
    use std::path::PathBuf;
    use std::process::Command;

    const D_MODEL: usize = 64;
    const D_FF: usize = 128;
    const N_LAYERS: usize = 2;
    const N_EXPERTS: usize = 2;

    fn build_synthetic_gguf(path: &PathBuf) {
        use candle_core::quantized::gguf_file::Value;

        let arch = Value::String("mixtral".to_string());
        let metadata: Vec<(&str, &Value)> = vec![("general.architecture", &arch)];

        let mut tensors: Vec<(String, QTensor)> = Vec::new();
        tensors.push(("tok_embeddings.weight".into(), qtensor_f32(&[32000usize, D_MODEL])));
        tensors.push(("output.weight".into(), qtensor_f32(&[32000, D_MODEL])));

        for layer in 0..N_LAYERS {
            tensors.push((format!("blk.{layer}.attn_q.weight"), qtensor_f32(&[D_MODEL, D_MODEL])));
            tensors.push((format!("blk.{layer}.attn_k.weight"), qtensor_f32(&[D_MODEL, D_MODEL])));
            tensors.push((format!("blk.{layer}.attn_v.weight"), qtensor_f32(&[D_MODEL, D_MODEL])));
            tensors.push((format!("blk.{layer}.attn_output.weight"), qtensor_f32(&[D_MODEL, D_MODEL])));
            tensors.push((format!("blk.{layer}.attn_norm.weight"), qtensor_f32(&[D_MODEL])));
            tensors.push((format!("blk.{layer}.ffn_norm.weight"), qtensor_f32(&[D_MODEL])));
            tensors.push((format!("blk.{layer}.ffn_gate_inp.weight"), qtensor_f32(&[N_EXPERTS, D_MODEL])));

            for expert in 0..N_EXPERTS {
                tensors.push((format!("blk.{layer}.ffn_gate.{expert}.weight"), qtensor_q4_0(&[D_FF, D_MODEL])));
                tensors.push((format!("blk.{layer}.ffn_up.{expert}.weight"), qtensor_q4_0(&[D_FF, D_MODEL])));
                tensors.push((format!("blk.{layer}.ffn_down.{expert}.weight"), qtensor_q4_0(&[D_MODEL, D_FF])));
            }
        }

        let tensor_pairs: Vec<(&str, &QTensor)> = tensors.iter().map(|(n, t)| (n.as_str(), t)).collect();
        let mut file = std::fs::File::create(path).unwrap();
        gguf_file::write(&mut file, &metadata, &tensor_pairs).unwrap();
    }

    fn qtensor_f32(shape: &[usize]) -> QTensor {
        let elems: usize = shape.iter().product();
        let data: Vec<f32> = (0..elems).map(|i| (i as f32 + 1.0) * 0.01).collect();
        QTensor::quantize(
            &candle_core::Tensor::from_vec(data, shape, &candle_core::Device::Cpu).unwrap(),
            GgmlDType::F32,
        )
        .unwrap()
    }

    fn qtensor_q4_0(shape: &[usize]) -> QTensor {
        let elems: usize = shape.iter().product();
        let data: Vec<f32> = (0..elems).map(|i| i as f32 * 0.01 - 0.5).collect();
        QTensor::quantize(
            &candle_core::Tensor::from_vec(data, shape, &candle_core::Device::Cpu).unwrap(),
            GgmlDType::Q4_0,
        )
        .unwrap()
    }

    #[test]
    fn extract_and_verify_synthetic_gguf() {
        let tmp = tempfile::TempDir::new().unwrap();
        let gguf_path = tmp.path().join("test.gguf");
        let out_dir = tmp.path().join("extracted");

        build_synthetic_gguf(&gguf_path);
        assert!(gguf_path.exists(), "synthetic GGUF not created");

        let binary = std::env::var("CARGO_BIN_EXE_mistralrs-expert-extract")
            .unwrap_or_else(|_| {
                std::env::current_dir()
                    .unwrap()
                    .join("target/debug/mistralrs-expert-extract")
                    .to_string_lossy()
                    .into_owned()
            });
        let status = Command::new(&binary)
            .arg("-i")
            .arg(&gguf_path)
            .arg("-o")
            .arg(&out_dir)
            .status()
            .unwrap();
        assert!(status.success(), "extraction failed");

        let manifest_path = out_dir.join("manifest.json");
        assert!(manifest_path.exists(), "manifest.json missing");
        let manifest_bytes = fs::read_to_string(&manifest_path).unwrap();
        let manifest: serde_json::Value = serde_json::from_str(&manifest_bytes).unwrap();
        assert_eq!(manifest["num_layers"].as_u64().unwrap(), N_LAYERS as u64);
        assert_eq!(manifest["num_experts_per_layer"].as_u64().unwrap(), N_EXPERTS as u64);
        assert_eq!(manifest["expert_map"].as_object().unwrap().len(), N_LAYERS * N_EXPERTS);

        for layer in 0..N_LAYERS {
            for exp in 0..N_EXPERTS {
                let global_id = (layer * N_EXPERTS + exp) as u64;
                let entry = &manifest["expert_map"][global_id.to_string()];
                assert_eq!(entry["layer_idx"].as_u64().unwrap(), layer as u64);
                assert_eq!(entry["local_id"].as_u64().unwrap(), exp as u64);
                assert_eq!(entry["d_model"].as_u64().unwrap(), D_MODEL as u64);
                assert_eq!(entry["d_ff"].as_u64().unwrap(), D_FF as u64);
                assert_eq!(entry["dtype"].as_str().unwrap(), "Q4_0");
            }
        }

        let experts_dir = out_dir.join("experts");
        assert!(experts_dir.is_dir());
        for layer in 0..N_LAYERS {
            for exp in 0..N_EXPERTS {
                let fname = format!("layer_{layer}_expert_{exp}.bin");
                let fpath = experts_dir.join(&fname);
                assert!(fpath.exists(), "missing {fname}");

                let meta = fs::metadata(&fpath).unwrap();
                let gate_up_bytes = 2 * D_FF * D_MODEL * 18 / 32;
                let down_bytes = D_MODEL * D_FF * 18 / 32;
                let expected = (gate_up_bytes + down_bytes) as u64;
                assert_eq!(meta.len(), expected, "wrong size for {fname}");

                let data = fs::read(&fpath).unwrap();
                assert!(data.iter().any(|&b| b != 0), "{fname} is all zeros");
            }
        }

        assert!(out_dir.join("dense.gguf").exists(), "dense.gguf missing");
    }
}
