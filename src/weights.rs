//! Checkpoint access: mmap the safetensors file, zero-copy.
//!
//! The omniASR-CTC checkpoint is stored f32; f16/bf16 tensors are accepted and
//! widened (exact) for the CPU path. GPU uploads go through `gpu`'s uploader.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use bytes::Bytes;
use memmap2::Mmap;
use safetensors::Dtype;

/// One tensor as it sits in the file.
#[derive(Debug, Clone)]
pub(crate) struct RawTensor {
    pub data: Bytes,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

impl RawTensor {
    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        Ok(match self.dtype {
            Dtype::F32 => self
                .data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            Dtype::F16 => self
                .data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half_from_le(c).to_f32())
                .collect(),
            Dtype::BF16 => self
                .data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f32::from_bits((u16::from_ne_bytes([c[0], c[1]]) as u32) << 16))
                .collect(),
            other => return Err(anyhow!("unsupported dtype {other:?} for to_f32_vec")),
        })
    }

    pub fn as_f32(&self) -> Result<(Vec<f32>, Vec<usize>)> {
        Ok((self.to_f32_vec()?, self.shape.clone()))
    }

    /// The tensor as little-endian `f32` bytes, ready for `write_buffer`.
    ///
    /// An `f32` checkpoint on a little-endian host is already exactly that,
    /// so the bytes are borrowed straight out of the mmap and nothing is
    /// copied: on this machine ~95% of the checkpoint reaches the GPU without
    /// ever touching the host heap. `f16`/`bf16` (and big-endian hosts) widen
    /// into an owned buffer, widened once and then dropped right after the
    /// upload.
    pub fn as_f32_le_bytes(&self) -> Result<Cow<'_, [u8]>> {
        #[cfg(target_endian = "little")]
        if self.dtype == Dtype::F32 {
            return Ok(Cow::Borrowed(&self.data));
        }
        let widened = self.to_f32_vec()?;
        Ok(Cow::Owned(
            widened.iter().flat_map(|f| f.to_le_bytes()).collect(),
        ))
    }
}

/// [`get_f32`] for callers that upload bytes: the tensor's little-endian
/// `f32` bytes plus its shape.
///
/// Borrowed from the mmap whenever the checkpoint is `f32`, which is what
/// keeps a 1.2 GiB weight upload free of host-side transposes and copies.
pub(crate) fn get_f32_bytes<'a>(
    w: &'a HashMap<String, RawTensor>,
    name: &str,
) -> Result<(Cow<'a, [u8]>, &'a [usize])> {
    let t = w
        .get(name)
        .ok_or_else(|| anyhow!("weight not found: {name}"))?;
    Ok((t.as_f32_le_bytes()?, &t.shape))
}

fn half_from_le(c: &[u8]) -> f16 {
    f16::from_le_bytes([c[0], c[1]])
}

/// Minimal IEEE 754 binary16 → f32 (exact); avoids a dependency for one call.
#[allow(non_camel_case_types)]
struct f16(u16);

impl f16 {
    fn from_le_bytes(b: [u8; 2]) -> Self {
        f16(u16::from_le_bytes(b))
    }
    fn to_f32(&self) -> f32 {
        let bits = self.0 as u32;
        let sign = (bits >> 15) & 1;
        let exp = (bits >> 10) & 0x1f;
        let frac = bits & 0x3ff;
        let f = if exp == 0 {
            if frac == 0 {
                sign << 31
            } else {
                // subnormal: normalise
                let mut e = 127u32 - 15 + 1;
                let mut fr = frac;
                while fr & 0x400 == 0 {
                    fr <<= 1;
                    e -= 1;
                }
                fr &= 0x3ff;
                (sign << 31) | (e << 23) | (fr << 13)
            }
        } else if exp == 0x1f {
            (sign << 31) | 0x7f80_0000u32 | (frac << 13)
        } else {
            (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
        };
        f32::from_bits(f)
    }
}

/// mmap every safetensors shard, zero-copy.
pub(crate) fn load_tensors(model_dir: &Path) -> Result<HashMap<String, RawTensor>> {
    let index = model_dir.join("model.safetensors.index.json");
    if index.exists() {
        let idx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&index)?)?;
        let wm = idx["weight_map"]
            .as_object()
            .ok_or_else(|| anyhow!("invalid index.json"))?;
        let mut shards: Vec<String> = wm
            .values()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect();
        shards.sort();
        shards.dedup();
        let mut all = HashMap::new();
        for s in shards {
            all.extend(load_shard(&model_dir.join(&s))?);
        }
        return Ok(all);
    }
    load_shard(&model_dir.join("model.safetensors"))
}

fn load_shard(path: &Path) -> Result<HashMap<String, RawTensor>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    // SAFETY: read-only checkpoint, never mutated while mapped.
    let mmap = unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", path.display()))?;
    let buf: Bytes = Bytes::from_owner(mmap);
    let st = safetensors::SafeTensors::deserialize(&buf)?;
    let base = buf.as_ptr() as usize;

    let mut out = HashMap::with_capacity(st.len());
    for (name, view) in st.iter() {
        let vd = view.data();
        let offset = vd.as_ptr() as usize - base;
        out.insert(
            name.to_string(),
            RawTensor {
                data: buf.slice(offset..offset + vd.len()),
                shape: view.shape().to_vec(),
                dtype: view.dtype(),
            },
        );
    }
    Ok(out)
}

pub(crate) fn get_f32(
    w: &HashMap<String, RawTensor>,
    name: &str,
) -> Result<(Vec<f32>, Vec<usize>)> {
    let t = w
        .get(name)
        .ok_or_else(|| anyhow!("weight not found: {name}"))?;
    t.as_f32()
}

#[cfg(test)]
mod tests {
    use super::{get_f32_bytes, RawTensor};
    use safetensors::Dtype;
    use std::collections::HashMap;

    fn tensor(dtype: Dtype, data: Vec<u8>) -> RawTensor {
        let n = data.len() / dtype.size();
        RawTensor {
            data: bytes::Bytes::from(data),
            shape: vec![n],
            dtype,
        }
    }

    /// `[1.0, -2.5, 0.25]` in the three dtypes the loader accepts.
    fn f32_le() -> Vec<u8> {
        [1.0f32, -2.5, 0.25]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect()
    }
    fn f16_le() -> Vec<u8> {
        [0x3c00u16, 0xc100, 0x3400]
            .iter()
            .flat_map(|b| b.to_le_bytes())
            .collect()
    }
    fn bf16_le() -> Vec<u8> {
        [0x3f80u16, 0xc020, 0x3e80]
            .iter()
            .flat_map(|b| b.to_le_bytes())
            .collect()
    }

    #[test]
    fn f32_bytes_borrow_the_mmap_and_narrower_dtypes_widen() {
        let want = [1.0f32, -2.5, 0.25];

        let t = tensor(Dtype::F32, f32_le());
        let bytes = t.as_f32_le_bytes().unwrap();
        // zero-copy: the bytes are the mmap slice itself, not a copy of it
        #[cfg(target_endian = "little")]
        assert!(
            std::ptr::eq(bytes.as_ptr(), t.data.as_ptr()),
            "an f32 tensor must be borrowed in place"
        );
        let got: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(got, want);

        for dtype in [Dtype::F16, Dtype::BF16] {
            let raw = if dtype == Dtype::F16 {
                f16_le()
            } else {
                bf16_le()
            };
            let t = tensor(dtype, raw);
            let bytes = t.as_f32_le_bytes().unwrap();
            assert!(
                matches!(bytes, std::borrow::Cow::Owned(_)),
                "{dtype:?} must widen, not borrow"
            );
            let got: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(got, want, "{dtype:?}");
        }
    }

    #[test]
    fn get_f32_bytes_returns_the_shape_alongside_the_bytes() {
        let w: HashMap<String, RawTensor> = [("w".to_string(), tensor(Dtype::F32, f32_le()))]
            .into_iter()
            .collect();
        let (bytes, shape) = get_f32_bytes(&w, "w").unwrap();
        assert_eq!(bytes.len(), 12);
        assert_eq!(shape, &[3]);
        assert!(get_f32_bytes(&w, "missing").is_err());
    }
}
