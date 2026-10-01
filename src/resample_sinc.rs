//! torchaudio `sinc_interp_hann` resampling, ported line-for-line from
//! `torchaudio/functional/functional.py` (`_get_sinc_resample_kernel` +
//! `_apply_sinc_resample_kernel`) so the Rust side consumes the same waveform
//! the Python reference does.

/// Resample `input` from `orig_freq` to `new_freq` (defaults: width 6,
/// rolloff 0.99, Hann window).
pub fn resample(input: &[f32], orig_freq: u32, new_freq: u32) -> Vec<f32> {
    const LOWPASS_WIDTH: f32 = 6.0;
    const ROLLOFF: f32 = 0.99;

    if orig_freq == new_freq {
        return input.to_vec();
    }
    let gcd = gcd(orig_freq, new_freq);
    let orig_freq = (orig_freq / gcd) as f32;
    let new_freq = (new_freq / gcd) as f32;
    let new_freq_i = new_freq as usize;

    let base_freq = orig_freq.min(new_freq) * ROLLOFF;
    let width = (LOWPASS_WIDTH * orig_freq / base_freq).ceil() as i64;

    // kernel: one row per output phase (new_freq rows), 2*width+orig taps
    let taps = (2 * width + orig_freq as i64) as usize;
    let mut kernel = vec![0.0f32; new_freq_i * taps];
    for j in 0..new_freq_i {
        for (ti, item) in ((-width)..(width + orig_freq as i64)).enumerate() {
            // t = -j / new_freq + idx / orig_freq, scaled by base_freq
            let mut t = (item as f32 / orig_freq) - (j as f32 / new_freq);
            t *= base_freq;
            t = t.clamp(-LOWPASS_WIDTH, LOWPASS_WIDTH);
            let window = (t * std::f32::consts::PI / LOWPASS_WIDTH / 2.0).cos().powi(2);
            let t = t * std::f32::consts::PI;
            let sinc = if t == 0.0 { 1.0 } else { t.sin() / t };
            kernel[j * taps + ti] = sinc * window * (base_freq / orig_freq);
        }
    }

    // conv1d(stride=orig_freq) over x padded with (width, width + orig_freq)
    let in_len = input.len() as i64;
    let frames = ((in_len + width + width + orig_freq as i64 - taps as i64) / orig_freq as i64
        + 1) as usize;
    let target_len = ((new_freq * in_len as f32) / orig_freq).ceil() as usize;
    let mut out = vec![0.0f32; target_len];

    for m in 0..target_len {
        let j = m % new_freq_i; // output phase
        let f = m / new_freq_i; // conv frame
        if f >= frames {
            break;
        }
        let base = f as i64 * orig_freq as i64 - width; // index into `input` of tap 0
        let row = &kernel[j * taps..(j + 1) * taps];
        let mut acc = 0.0f32;
        for (ti, k) in row.iter().enumerate() {
            let idx = base + ti as i64;
            if idx >= 0 && idx < in_len {
                acc += input[idx as usize] * k;
            }
        }
        out[m] = acc;
    }
    out
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn length_matches_torchaudio_formula() {
        // 24000 -> 16000: gcd 8000, orig 3, new 2, width 10
        let out = super::resample(&vec![0.0f32; 16000], 24000, 16000);
        let target = ((2.0f32 * 16000.0) / 3.0).ceil() as usize;
        assert_eq!(out.len(), target);
    }
}
