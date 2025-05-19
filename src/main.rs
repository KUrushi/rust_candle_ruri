mod ruri;

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, };
use candle_core;
use candle_core::Device::Cpu;
use candle_nn::VarBuilder;
use ruri as modernbert;
use clap::{Parser, ValueEnum};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{PaddingParams, Tokenizer};
use candle_core::{Device, Tensor, Result, DType, Error as E, D};
use safetensors::{serialize_to_file};

fn main() -> anyhow::Result<()> {
    let model_id = "cl-nagoya/ruri-v3-310m";
    let revision = "main";

    let repo = Repo::with_revision(model_id.to_string(),
                                   RepoType::Model,
                                   revision.to_string().to_owned());

    let api = Api::new()?;
    let api = api.repo(repo);
    let config = api.get("config.json")?;
    let tokenizer = api.get("tokenizer.json")?;
    let weights = api.get("model.safetensors")?;
    println!("{:?}", weights);
    let config = std::fs::read_to_string(config)?;
    let config: modernbert::Config = serde_json::from_str(&config)?;
    let mut tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights], candle_core::DType::F32, &Cpu).unwrap()
    };

    tokenizer.with_padding(Some(PaddingParams {
        strategy: tokenizers::PaddingStrategy::BatchLongest,
        pad_id: config.pad_token_id,
        ..Default::default()
    }))
        .with_truncation(None).map_err(E::msg)?;


    let model = modernbert::ModernBert::load(vb, &config)?;

    let prompt = vec!["こんにちは!"];
    let input_ids = tokenize_batch(&tokenizer, prompt.clone(), &Cpu)?;
    let attention_mask = get_attention_mask(&tokenizer, prompt.clone(), &Cpu)?;
    let output = model.forward(&input_ids, &attention_mask)?
        .to_dtype(candle_core::DType::F32)?;
    let output = mean_pooling(&output, &attention_mask)?;


    let mut tensors_to_save = HashMap::new();
    tensors_to_save.insert("test".to_string(), output);
    let path = Path::new("sample_tensors.safetensors");
    serialize_to_file(&tensors_to_save, &None, path)?;

    Ok(())
}

pub fn mean_pooling(
    token_embeddings: &Tensor,
    attention_mask: &Tensor,
) -> Result<Tensor> {
    let device = token_embeddings.device();

    // 1. アテンションマスクを拡張し、float型に変換
    // attention_mask の形状は (batch_size, sequence_length)
    // これを (batch_size, sequence_length, 1) に unsqueeze し、乗算と合計のために DType::F32 に変換
    let attention_mask_expanded = attention_mask
        .to_dtype(DType::F32)?
        .unsqueeze(D::Minus1)?; // (batch_size, sequence_length, 1)

    // 2. トークン埋め込みをマスキング
    // token_embeddings (B, S, E) と attention_mask_expanded (B, S, 1) を乗算
    // attention_mask_expanded は embedding_dimension に沿ってブロードキャストされる
    let masked_embeddings = token_embeddings.broadcast_mul(&attention_mask_expanded)?;

    // 3. マスキングされた埋め込みを sequence_length 次元 (dim=1) に沿って合計
    // 結果の形状: (batch_size, 1, embedding_dimension)
    let sum_embeddings = masked_embeddings.sum_keepdim(1)?;

    // 4. アテンションマスクを sequence_length 次元 (dim=1) に沿って合計し、トークン数を取得
    // attention_mask_expanded は (batch_size, sequence_length, 1)
    // これを合計すると、各文の実際のトークン数が得られる (形状: batch_size, 1, 1)
    let sum_mask = attention_mask_expanded.sum_keepdim(1)?; // (batch_size, 1, 1)

    // 5. ゼロ除算を避けるために sum_mask をクランプ（最小値を設定）
    // PyTorch の torch.clamp(sum_mask, min=1e-9) と同様の処理
    let min_val_for_clamp = 1e-9f32;
    // sum_mask と同じ形状で、値が min_val_for_clamp のテンソルを作成
    let clamp_tensor = Tensor::full(min_val_for_clamp, sum_mask.shape(), device)?;
    let sum_mask_clamped = sum_mask.maximum(&clamp_tensor)?;


    // 6. 合計された埋め込みを、クランプされた合計マスクで除算
    // sum_embeddings: (batch_size, 1, embedding_dimension)
    // sum_mask_clamped: (batch_size, 1, 1)
    // 除算により sum_mask_clamped がブロードキャストされる
    let mean_pooled_embeddings = sum_embeddings.broadcast_div(&sum_mask_clamped)?;

    // 結果の形状は (batch_size, 1, embedding_dimension) なので、
    // 通常望まれる (batch_size, embedding_dimension) にするために次元1をsqueezeする
    mean_pooled_embeddings.squeeze(1)
}
fn get_attention_mask(tokenizer: &Tokenizer, input: Vec<&str>, device: &Device) -> anyhow::Result<Tensor> {
    let tokens = tokenizer.encode_batch(input, true).map_err(E::msg)?;
    let attention_mask = tokens.iter().map(|tokens| {
        let tokens = tokens.get_attention_mask().to_vec();
        Tensor::new(tokens.as_slice(), device)
    }).collect::<candle_core::Result<Vec<_>>>()?;
    Ok(Tensor::stack(&attention_mask, 0)?)
}
fn tokenize_batch(
    tokenizer: &Tokenizer,
    input: Vec<&str>,
    device: &Device,
) -> anyhow::Result<Tensor> {
    let tokens = tokenizer.encode_batch(input, true).map_err(E::msg)?;
    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Tensor::new(tokens.as_slice(), device)
        }).collect::<candle_core::Result<Vec<_>>>()?;
    Ok(Tensor::stack(&token_ids, 0)?)
}