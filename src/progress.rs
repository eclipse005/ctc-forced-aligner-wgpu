//! Progress for one alignment run.
//!
//! One counter spans encode, Viterbi, traceback, the score replay, and the
//! timeline. It starts at 0 and reaches 100 on the finish tick.
//!
//! A tick is a finished checkpoint: rows back in RAM, or a traceback segment
//! whose choices have been walked. Work still queued on the device is not a
//! tick. The denominator is set before the first tick. If a backpointer
//! reserve fails after that, the extra traceback segments are added before
//! they tick. A percentage already delivered can then be larger than the next one.
//!
//! One unit is one window of the file. A phase that walks the whole file
//! spends that many units; the traceback spends one per segment it rebuilds.

use std::sync::Mutex;

/// Which part of the run is moving. The stages run in this order; a phase
/// that is fused with another (the GPU window loop encodes and runs the DP on
/// the same queue) reports as [`Stage::Encode`] because that is when its
/// window is finished.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// A window's encoder rows and DP choices, finished.
    Encode,
    /// The Viterbi's own forward pass, on the paths that run it separately
    /// from the encoder (the CPU tower, the gathered trellis).
    Dp,
    /// Rebuilding or walking the choices back to frame 0.
    Traceback,
    /// Replaying the lm head over the traced path to score it.
    Scores,
    /// `collapse` → tokens → words → segments.
    Finish,
}

impl Stage {
    /// A short fixed-width label, for a one-line display.
    pub fn label(self) -> &'static str {
        match self {
            Stage::Encode => "encode",
            Stage::Dp => "viterbi",
            Stage::Traceback => "traceback",
            Stage::Scores => "scores",
            Stage::Finish => "finish",
        }
    }
}

/// One progress tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Progress {
    /// The phase that produced this tick.
    pub stage: Stage,
    /// Units finished so far, across every phase. Monotone for the run.
    pub done: usize,
    /// Units the whole run has. Never 0 for a run that reports.
    pub total: usize,
}

impl Progress {
    /// `done / total` clamped to 0..=1. 0 when there is nothing to count.
    pub fn fraction(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        (self.done.min(self.total) as f64 / self.total as f64).clamp(0.0, 1.0)
    }

    /// [`fraction`](Self::fraction) as a whole percentage, floored and clamped
    /// to 0..=100. Integer arithmetic: the value a terminal line wants, and
    /// the same one the CLI's own tests pin.
    pub fn pct(&self) -> u8 {
        if self.total == 0 {
            return 0;
        }
        let done = self.done.min(self.total) as u128;
        (done * 100 / self.total as u128).min(100) as u8
    }
}

/// Where a caller hangs its progress sink.
///
/// `Fn + Send + Sync`, not `FnMut`: the encoder thread and the DP worker both
/// fire, and the callback runs under one lock so a later tick cannot be
/// delivered first. The sink must return without waiting for the caller
/// thread, and must not call back into this run's progress. A channel send
/// is the safe shape.
pub type AlignProgress<'a> = &'a (dyn Fn(Progress) + Send + Sync);

struct Counters {
    done: usize,
    total: usize,
}

/// The run's counter and its sink. One `done` for the whole run.
///
/// The lock covers the counter and the callback together. Two threads can
/// fire; the callback sees `done` in the order the counter moved.
pub(crate) struct ProgressState<'a> {
    sink: Option<AlignProgress<'a>>,
    inner: Mutex<Counters>,
}

impl<'a> ProgressState<'a> {
    pub(crate) fn new(sink: Option<AlignProgress<'a>>) -> Self {
        Self { sink, inner: Mutex::new(Counters { done: 0, total: 0 }) }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Counters> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Publish the run's denominator. Called before the first tick.
    pub(crate) fn set_total(&self, total: usize) {
        self.lock().total = total;
    }

    /// Add traceback units discovered after the denominator was published.
    ///
    /// Called when the full choice table fits the budget on paper but the
    /// reserve fails, and only before the ticks that spend the new units.
    /// A callback that already went out still carries the old total, so the
    /// next fraction can be smaller once.
    pub(crate) fn grow_total(&self, extra: usize) {
        if extra == 0 {
            return;
        }
        let mut g = self.lock();
        g.total = g.total.saturating_add(extra);
    }

    /// Units finished so far.
    pub(crate) fn done(&self) -> usize {
        self.lock().done
    }

    /// Units the run promised.
    pub(crate) fn total(&self) -> usize {
        self.lock().total
    }

    /// Record `units` finished checkpoints and, when a sink is set, deliver them.
    ///
    /// The counter moves even when there is no sink. The end-of-run check
    /// reads `done`, and a debug `align(..., None)` is the path the tests take.
    pub(crate) fn fire(&self, stage: Stage, units: usize) {
        if units == 0 {
            return;
        }
        let mut g = self.lock();
        g.done += units;
        // A phase that reports more than the run predicted must not draw a
        // fraction past 1. The stored denominator stays what the run published.
        let total = g.total.max(g.done);
        let done = if stage == Stage::Finish { total } else { g.done };
        if let Some(sink) = self.sink {
            sink(Progress { stage, done, total });
        }
    }
}

/// The Viterbi's own progress: frames in, units out.
///
/// A phase that walks the whole file spends `n_windows` units, so the DP
/// counts frames and converts against that quantum. `None` on the fused GPU
/// path, where the DP rides inside the encoder's window: there the window
/// tick already paid for those frames, and counting them twice would run the
/// bar ahead of the wall clock and then strand it.
pub(crate) struct DpProgress<'a, 'p> {
    state: &'a ProgressState<'p>,
    /// Units this phase owns, and how many it has fired.
    units: usize,
    fired: usize,
    /// Frames per unit, and frames seen so far.
    quantum: usize,
    frames: usize,
}

impl<'a, 'p> DpProgress<'a, 'p> {
    /// `units` checkpoints for a DP over `frames` frames.
    pub(crate) fn new(state: &'a ProgressState<'p>, units: usize, frames: usize) -> Self {
        Self {
            state,
            units,
            fired: 0,
            quantum: frames.div_ceil(units.max(1)).max(1),
            frames: 0,
        }
    }

    /// `frames` more frames of the forward pass have been computed.
    pub(crate) fn step(&mut self, frames: usize) {
        if self.units == 0 {
            return;
        }
        self.frames += frames;
        while self.fired < self.units && self.frames >= (self.fired + 1) * self.quantum {
            self.fired += 1;
            self.state.fire(Stage::Dp, 1);
        }
    }

    /// The forward pass is over: spend whatever the frames did not reach, so
    /// the phase lands exactly on its own unit count.
    pub(crate) fn finish(&mut self) {
        if self.units > self.fired {
            let left = self.units - self.fired;
            self.fired = self.units;
            self.state.fire(Stage::Dp, left);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Progress, ProgressState, Stage};
    use std::sync::Arc;

    fn pct_of(done: usize, total: usize) -> u8 {
        Progress { stage: Stage::Encode, done, total }.pct()
    }

    #[test]
    fn a_full_run_ends_at_one_hundred_exactly() {
        assert_eq!(pct_of(0, 124), 0);
        assert_eq!(pct_of(124, 124), 100);
    }

    /// The unit count has to survive the one place it can go wrong: a phase
    /// that reports more than the run predicted must not draw past the end.
    #[test]
    fn over_reporting_clamps_instead_of_overshooting() {
        assert_eq!(pct_of(9, 3), 100);
    }

    /// No denominator is not a fake single hop to 100%: an empty run reports
    /// nothing at all, and a tick that arrives anyway reads 0.
    #[test]
    fn no_denominator_is_zero_not_a_fake_hop() {
        assert_eq!(pct_of(1, 0), 0);
        assert_eq!(Progress { stage: Stage::Encode, done: 1, total: 0 }.fraction(), 0.0);
    }

    #[test]
    fn the_counter_is_monotone_across_phases_and_keeps_one_denominator() {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let tap = seen.clone();
        let sink = move |p: Progress| tap.lock().unwrap().push((p.stage, p.done, p.total));
        let state = ProgressState::new(Some(&sink));
        state.set_total(10);
        state.fire(Stage::Encode, 6);
        state.fire(Stage::Traceback, 3);
        state.fire(Stage::Finish, 1);

        let got = seen.lock().unwrap().clone();
        assert_eq!(
            got,
            vec![
                (Stage::Encode, 6, 10),
                (Stage::Traceback, 9, 10),
                (Stage::Finish, 10, 10),
            ]
        );
        let pcts: Vec<u8> = got.iter().map(|&(_, d, t)| pct_of(d, t)).collect();
        assert_eq!(pcts, vec![60, 90, 100]);
        assert!(pcts.windows(2).all(|w| w[1] >= w[0]), "progress went backwards");
    }

    /// No sink skips the callback. The counter still moves: a debug run with
    /// `None` checks `done + 1 == total` after a successful alignment.
    #[test]
    fn no_sink_counts_and_still_meets_the_end_of_run_check() {
        let state = ProgressState::new(None);
        state.set_total(4);
        state.fire(Stage::Encode, 2);
        state.fire(Stage::Traceback, 1);
        assert_eq!(state.done() + 1, state.total());
        state.fire(Stage::Finish, 1);
        assert_eq!(state.done(), state.total());
    }

    /// The reserve failed after encode had already published a one-step
    /// traceback. The extra segments are added before they tick, and the run
    /// still lands on the new total.
    #[test]
    fn a_grown_traceback_still_lands_on_its_total() {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let tap = seen.clone();
        let sink = move |p: Progress| tap.lock().unwrap().push((p.done, p.total, p.pct()));
        let state = ProgressState::new(Some(&sink));
        state.set_total(8);
        state.fire(Stage::Encode, 6);
        state.grow_total(3);
        for _ in 0..4 {
            state.fire(Stage::Traceback, 1);
        }
        assert_eq!(state.done() + 1, state.total());
        state.fire(Stage::Finish, 1);
        assert_eq!(state.done(), state.total());
        let got = seen.lock().unwrap().clone();
        assert_eq!(got.last().map(|t| t.2), Some(100));
        assert!(got.windows(2).all(|w| w[1].0 >= w[0].0), "done went backwards: {got:?}");
    }

    /// Encoder thread and DP worker both fire. The lock keeps the delivered
    /// `done` non-decreasing.
    #[test]
    fn concurrent_ticks_stay_ordered() {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let tap = seen.clone();
        let sink = move |p: Progress| tap.lock().unwrap().push(p.done);
        let state = ProgressState::new(Some(&sink));
        state.set_total(200);
        std::thread::scope(|s| {
            for _ in 0..2 {
                let state = &state;
                s.spawn(move || {
                    for _ in 0..100 {
                        state.fire(Stage::Encode, 1);
                    }
                });
            }
        });
        let got = seen.lock().unwrap().clone();
        assert_eq!(got.len(), 200);
        assert!(got.windows(2).all(|w| w[1] >= w[0]), "delivered out of order: {got:?}");
        assert_eq!(*got.last().unwrap(), 200);
        assert_eq!(state.done(), 200);
    }

    #[test]
    fn a_zero_unit_tick_is_not_a_tick() {
        let seen = Arc::new(std::sync::Mutex::new(0usize));
        let tap = seen.clone();
        let sink = move |_: Progress| *tap.lock().unwrap() += 1;
        let state = ProgressState::new(Some(&sink));
        state.set_total(3);
        state.fire(Stage::Encode, 0);
        assert_eq!(*seen.lock().unwrap(), 0);
    }

    #[test]
    fn every_stage_has_a_label() {
        for stage in [Stage::Encode, Stage::Dp, Stage::Traceback, Stage::Scores, Stage::Finish] {
            assert!(!stage.label().is_empty());
        }
    }

    /// Frames convert to units against the phase's own budget, and the phase
    /// lands exactly on it: no drift to a bar that stops short of 100%.
    #[test]
    fn frames_land_exactly_on_the_phase_budget() {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let tap = seen.clone();
        let sink = move |p: Progress| tap.lock().unwrap().push(p.done);
        let state = ProgressState::new(Some(&sink));
        state.set_total(10);
        // 7 windows' worth of DP over 7000 frames: 1000 frames per unit.
        let mut dp = super::DpProgress::new(&state, 7, 7000);
        for _ in 0..7 {
            dp.step(1000);
        }
        // 999 more frames than the budget asked for must not overshoot it.
        dp.step(999);
        dp.finish();
        assert_eq!(*seen.lock().unwrap(), vec![1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(state.done(), 7);
    }

    /// `units == 0` is the fused GPU path's DP: its frames are the encoder
    /// window's, already counted there.
    #[test]
    fn a_zero_unit_dp_phase_spends_nothing() {
        let seen = Arc::new(std::sync::Mutex::new(0usize));
        let tap = seen.clone();
        let sink = move |_: Progress| *tap.lock().unwrap() += 1;
        let state = ProgressState::new(Some(&sink));
        state.set_total(4);
        let mut dp = super::DpProgress::new(&state, 0, 7000);
        dp.step(7000);
        dp.finish();
        assert_eq!(*seen.lock().unwrap(), 0);
        assert_eq!(state.done(), 0);
    }
}