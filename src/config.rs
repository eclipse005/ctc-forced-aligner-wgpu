//! `config.json` of the omniASR-CTC checkpoint, narrowed to what the forward
//! pass needs.

use anyhow::{bail, Context, Result};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub(crate) struct Wav2Vec2Config {
    pub conv_dim: Vec<usize>,
    pub conv_kernel: Vec<usize>,
    pub conv_stride: Vec<usize>,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub num_conv_pos_embeddings: usize,
    pub num_conv_pos_embedding_groups: usize,
    pub vocab_size: usize,
    pub layer_norm_eps: f64,
    pub do_stable_layer_norm: bool,
    pub feat_extract_norm: String,
    #[allow(dead_code)]
    pub feat_extract_activation: String,
    #[allow(dead_code)]
    pub hidden_act: String,
    /// The CTC blank. `Wav2Vec2ForCTC` has no separate blank id — the pad token
    /// fills that role, which is why the reference reads `pad_token_id`.
    pub pad_token_id: usize,
}

impl Wav2Vec2Config {
    pub fn load(model_dir: &std::path::Path) -> Result<Self> {
        let cfg: Wav2Vec2Config = serde_json::from_str(
            &std::fs::read_to_string(model_dir.join("config.json"))
                .context("read config.json")?,
        )
        .context("parse config.json")?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        // The port implements exactly this architecture; anything else must
        // refuse rather than silently misalign.
        if !self.do_stable_layer_norm {
            bail!("only do_stable_layer_norm=true checkpoints are supported");
        }
        if self.feat_extract_norm != "layer" {
            bail!("only feat_extract_norm=\"layer\" is supported");
        }
        if self.feat_extract_activation != "gelu" || self.hidden_act != "gelu" {
            bail!("only gelu activations are supported");
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            bail!("hidden_size not divisible by heads");
        }
        Ok(())
    }

    /// Input samples per output frame: the product of the feature extractor's
    /// conv strides. The reference reads the same number as
    /// `config.inputs_to_logits_ratio`, and every timestamp in the output is
    /// `frame_rate = TARGET_SR / subsampling` seconds per frame — so a constant
    /// here would silently mis-time every other checkpoint rather than fail.
    pub fn subsampling(&self) -> usize {
        self.conv_stride.iter().product::<usize>().max(1)
    }

    /// Timestamps per second, i.e. `TARGET_SR / subsampling`.
    pub fn frame_rate(&self, target_sr: u32) -> f64 {
        target_sr as f64 / self.subsampling() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(strides: &[usize]) -> Wav2Vec2Config {
        Wav2Vec2Config {
            conv_dim: vec![],
            conv_kernel: vec![],
            conv_stride: strides.to_vec(),
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            num_conv_pos_embeddings: 128,
            num_conv_pos_embedding_groups: 16,
            vocab_size: 10288,
            layer_norm_eps: 1e-5,
            do_stable_layer_norm: true,
            feat_extract_norm: "layer".to_string(),
            feat_extract_activation: "gelu".to_string(),
            hidden_act: "gelu".to_string(),
            pad_token_id: 0,
        }
    }

    #[test]
    fn subsampling_is_the_conv_stride_product() {
        // The checkpoint this port was built against: 5*2*2*2*2*2*2 = 320.
        assert_eq!(cfg(&[5, 2, 2, 2, 2, 2, 2]).subsampling(), 320);
        assert_eq!(cfg(&[5, 2, 2, 2, 2, 2, 2]).frame_rate(16_000), 50.0);
        // A different feature extractor must not inherit 320.
        assert_eq!(cfg(&[4, 2, 2]).subsampling(), 16);
        assert_eq!(cfg(&[4, 2, 2]).frame_rate(16_000), 1000.0);
        assert_eq!(cfg(&[10]).frame_rate(16_000), 1600.0);
        // Degenerate config must not divide by zero.
        assert_eq!(cfg(&[0, 0]).subsampling(), 1);
    }
}
