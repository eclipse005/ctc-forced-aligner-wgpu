//! Counting global allocator: live byte counter + high-water mark, so the
//! profiler can answer "is that RSS growth live data or allocator slop".
//!
//! `CTC_ALLOC_STATS=1` makes [`crate::wav2vec2::prof::dump`] print the peak.
//! The counters are one relaxed atomic add/sub per allocation — noise for the
//! forward pass (~1e3 allocations per 34 s chunk), not for the GEMMs.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

pub struct Stats;

static LIVE: AtomicI64 = AtomicI64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);
/// live bytes per power-of-two size bucket (bucket k = sizes [2^k, 2^(k+1)))
static BUCKETS: [AtomicI64; 48] = [const { AtomicI64::new(0) }; 48];

#[inline]
fn bucket_of(size: usize) -> usize {
    (usize::BITS - 1 - size.max(1).leading_zeros()) as usize
}

unsafe impl GlobalAlloc for Stats {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() {
            let size = layout.size() as i64;
            let now = LIVE.fetch_add(size, Ordering::Relaxed) + size;
            PEAK.fetch_max(now.max(0) as u64, Ordering::Relaxed);
            BUCKETS[bucket_of(layout.size())].fetch_add(size, Ordering::Relaxed);
        }
        p
    }

    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        let size = layout.size() as i64;
        LIVE.fetch_sub(size, Ordering::Relaxed);
        BUCKETS[bucket_of(layout.size())].fetch_sub(size, Ordering::Relaxed);
        System.dealloc(p, layout)
    }

    unsafe fn realloc(&self, p: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let q = System.realloc(p, layout, new_size);
        if !q.is_null() {
            let delta = new_size as i64 - layout.size() as i64;
            let now = LIVE.fetch_add(delta, Ordering::Relaxed) + delta;
            PEAK.fetch_max(now.max(0) as u64, Ordering::Relaxed);
            BUCKETS[bucket_of(layout.size())].fetch_sub(layout.size() as i64, Ordering::Relaxed);
            BUCKETS[bucket_of(new_size)].fetch_add(new_size as i64, Ordering::Relaxed);
        }
        q
    }
}

/// (live bytes, peak bytes) since process start.
pub fn stats() -> (i64, u64) {
    (
        LIVE.load(Ordering::Relaxed),
        PEAK.load(Ordering::Relaxed),
    )
}

/// The size buckets holding live bytes, descending, above `min_mb`.
pub fn buckets(min_mb: u64) -> Vec<(usize, i64)> {
    (0..BUCKETS.len())
        .map(|k| (k, BUCKETS[k].load(Ordering::Relaxed)))
        .filter(|(_, b)| *b > (min_mb << 20) as i64)
        .collect()
}

/// True when `CTC_ALLOC_STATS` selects allocation reporting.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CTC_ALLOC_STATS").map(|v| v == "1").unwrap_or(false))
}
