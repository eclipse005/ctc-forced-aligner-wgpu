//! Character vocabulary of the omniASR-CTC checkpoint (vocab.json).
//!
//! blank = id 0 (`<s>`), unk = `<unk>`; tokenisation is per character and
//! skips out-of-vocabulary characters, mirroring the Python `_tokenise`, so
//! every timestamp maps to one visible character.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;

pub struct Vocab {
    pub char_to_id: HashMap<char, usize>,
    pub unk_id: usize,
    pub size: usize,
}

impl Vocab {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let raw: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(model_dir.join("vocab.json")).context("read vocab.json")?,
        )
        .context("parse vocab.json")?;
        let mut char_to_id = HashMap::new();
        let mut size = 0usize;
        let mut unk_id = 3usize;
        for (tok, id) in raw.as_object().context("vocab.json object")? {
            let id = id.as_u64().context("vocab id")? as usize;
            size = size.max(id + 1);
            if tok == "<unk>" {
                unk_id = id;
            }
            let mut chars = tok.chars();
            if let (Some(c), None) = (chars.next(), chars.next()) {
                char_to_id.insert(c, id);
            }
        }
        Ok(Self { char_to_id, unk_id, size })
    }

    pub fn tokenise(&self, text: &str) -> (Vec<usize>, Vec<String>) {
        let mut ids = Vec::new();
        let mut pieces = Vec::new();
        for c in text.chars() {
            if let Some(&id) = self.char_to_id.get(&c) {
                if id == self.unk_id {
                    continue;
                }
                ids.push(id);
                pieces.push(c.to_string());
            }
        }
        (ids, pieces)
    }
}
