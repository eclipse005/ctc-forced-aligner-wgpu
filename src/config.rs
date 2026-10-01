//! `config.json` of the omniASR-CTC checkpoint, narrowed to what the forward
//! pass needs.

use anyhow::{bail, Context, Result};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Wav2Vec2Config {
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
}
