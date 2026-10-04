//! The one place a timestamp is decided after the Viterbi path is known.
//!
//! A CTC path says which frames each character sat on. Four rules then turn
//! that into the spans a subtitle shows, and they used to live in four files
//! with the order between them written down in comments and in the order of
//! statements in `Aligner::align`. This module is that list, in order, with
//! each rule carrying the evidence for it.
//!
//! Nothing else may write a `start` or an `end` after the DP. A rule that
//! needs a new time comes here, and the pipeline below is where it goes —
//! which is the only way the five views of an alignment (`tokens`,
//! `segments`, `spans`, the SRT, the karaoke ASS) can be guaranteed to agree.
//!
//! The order matters and is not arbitrary:
//!
//! 1. `pad_into_silence`  a boundary sits in the middle of the pause beside
//!    it, with a bound on how much silence one token may claim — and a pause
//!    longer than [`crate::timeline::pad::MAX_PAUSE_SEC`] is silence, which
//!    belongs to nobody: the word starts where its own evidence starts. It
//!    runs first because everything after it reads the padded ends.
//! 2. `anchor_marks`      a mark has no phone, so it becomes a point at the
//!    end of the sound before it — or at the start of the sound after it when
//!    no sound precedes, the mirror case for stream-opening marks. It must
//!    run before the unmeasured characters are inserted, or a mark would
//!    become their anchor instead of the speech that actually carries time.
//! 3. `place_unmeasured`  a character the vocabulary had no target for lies
//!    between its placed neighbours. It runs last of the three because it is
//!    the only one that adds tokens, and it reads the spans the two above left.
//! 4. `round`             every number is rounded once, at the end, so no
//!    view can round a value another view already rounded.

pub(crate) use anchor::anchor_marks;
pub(crate) use pad::pad_into_silence;
pub(crate) use place::place_unmeasured;

pub(crate) mod anchor;
pub(crate) mod pad;
pub(crate) mod place;
