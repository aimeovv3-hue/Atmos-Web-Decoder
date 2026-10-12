//! IAMF on the IEC 61937 transport: a private burst type, for a sender and
//! a receiver that are both ours (Omniphony `docs/iamf-transport-plan.md`).
//!
//! The IEC assigned IAMF no data type, so no third-party equipment knows
//! this one. A burst's payload is whole OBUs, `Pd` counting its bytes: the
//! OBUs of one temporal unit, preceded now and then by a copy of the
//! sequence's descriptors with `obu_redundant_copy` set, so a receiver that
//! joins mid-stream opens the sequence at the next copy. A sequence header
//! without the flag starts a new sequence, as on the raw transport.
//!
//! Unlike a raw stream, a burst is a resync point: whatever a damaged burst
//! left unframed is dropped when the next one arrives, instead of swallowing
//! it.

use bridge_api::RPushResult;
use bridge_common::logging::bridge_log;
use bridge_common::shared::{AfterPush, SharedState};

use crate::{IamfState, push_iamf};

/// The private IEC 61937 data type (`Pc` bits 0-4) of an IAMF burst. Kept in
/// step with `spdif::DATA_TYPE_IAMF_PRIVATE` in Omniphony, which the sender
/// and the host's carrier table use.
pub const IAMF_DATA_TYPE: u8 = 0x1D;

/// The IEC 61937 data type carrying IAMF.
pub fn accepts_data_type(data_type: u8) -> bool {
    data_type == IAMF_DATA_TYPE
}

/// Decode one burst payload: whole OBUs, nothing carried over from the
/// burst before.
pub fn push_burst(
    state: &mut IamfState,
    shared: &mut SharedState,
    payload: &[u8],
    result: &mut RPushResult,
) -> AfterPush {
    // A burst never continues the one before: what that one left unframed
    // was cut short on the carrier.
    state.buf.clear();
    let after = push_iamf(state, shared, payload, result);
    if !state.buf.is_empty() {
        if let Some(bursts) = state.truncated_bursts.note() {
            bridge_log!(
                log::Level::Warn,
                "iamf: IEC 61937 burst ends inside an OBU ({} byte(s) dropped, {bursts} burst(s) so far)",
                state.buf.len()
            );
        }
        state.buf.clear();
    }
    after
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ObuKind, frame_obu, obu_kind};
    use abi_stable::std_types::{RString, RVec};
    use bridge_api::RDecodedFrame;

    /// `obu_redundant_copy` in an OBU header.
    const REDUNDANT: u8 = 0x04;

    struct Harness {
        state: IamfState,
        shared: SharedState,
    }

    impl Harness {
        fn new() -> Self {
            Self {
                state: IamfState::default(),
                shared: SharedState::new(false),
            }
        }

        fn result() -> RPushResult {
            RPushResult {
                frames: RVec::new(),
                error_message: RString::new(),
                did_reset: false,
            }
        }

        fn raw(&mut self, data: &[u8]) -> RPushResult {
            let mut result = Self::result();
            let _ = push_iamf(&mut self.state, &mut self.shared, data, &mut result);
            result
        }

        fn burst(&mut self, payload: &[u8]) -> RPushResult {
            let mut result = Self::result();
            let _ = push_burst(&mut self.state, &mut self.shared, payload, &mut result);
            result
        }

        fn bursts(&mut self, bursts: &[Vec<u8>]) -> Vec<RDecodedFrame> {
            let mut frames = Vec::new();
            for payload in bursts {
                let result = self.burst(payload);
                assert!(result.error_message.is_empty(), "{}", result.error_message);
                frames.extend(result.frames);
            }
            frames
        }
    }

    fn vector(name: &str) -> Option<Vec<u8>> {
        let dir = std::env::var_os("HARLETTY_IAMF_VECTORS")?;
        std::fs::read(std::path::Path::new(&dir).join(name)).ok()
    }

    fn pcm(frames: &[RDecodedFrame]) -> Vec<i32> {
        frames.iter().flat_map(|f| f.pcm.iter().copied()).collect()
    }

    /// A raw sequence as a sender puts it on the carrier: its descriptors,
    /// then its data OBUs `per_burst` to a burst, whole.
    struct Carried {
        descriptors: Vec<u8>,
        units: Vec<Vec<u8>>,
    }

    impl Carried {
        fn of(stream: &[u8], per_burst: usize) -> Self {
            let mut descriptors = Vec::new();
            let mut data = Vec::new();
            let mut pos = 0;
            while let Some(frame) = frame_obu(&stream[pos..]).unwrap() {
                let obu = &stream[pos..pos + frame.len];
                match obu_kind(frame.obu_type) {
                    ObuKind::SequenceHeader | ObuKind::Descriptor if data.is_empty() => {
                        descriptors.extend_from_slice(obu);
                    }
                    ObuKind::Data => data.push(obu.to_vec()),
                    _ => {}
                }
                pos += frame.len;
            }
            assert_eq!(pos, stream.len(), "a whole number of OBUs");
            Self {
                descriptors,
                units: data.chunks(per_burst).map(<[_]>::concat).collect(),
            }
        }

        /// The descriptors as the copy a sender repeats.
        fn redundant_copy(&self) -> Vec<u8> {
            let mut copy = self.descriptors.clone();
            let mut pos = 0;
            while let Some(frame) = frame_obu(&copy[pos..]).unwrap() {
                copy[pos] |= REDUNDANT;
                pos += frame.len;
            }
            copy
        }
    }

    // Sequence header: type 31, six bytes, `iamf` and two profiles.
    const HEADER: [u8; 8] = [31 << 3, 0x06, b'i', b'a', b'm', b'f', 0x00, 0x00];

    #[test]
    fn only_the_private_type_is_taken() {
        for data_type in 0..=u8::MAX {
            assert_eq!(accepts_data_type(data_type), data_type == 0x1D);
        }
    }

    /// A burst cut short on the carrier does not swallow the next one: what
    /// it left unframed goes when the next burst arrives.
    #[test]
    fn a_truncated_burst_does_not_reach_into_the_next() {
        let mut bridge = Harness::new();
        // An audio frame OBU announcing 100 bytes, 3 of them there.
        bridge.burst(&[6 << 3, 100, 0xAA, 0xBB, 0xCC]);
        assert!(bridge.state.buf.is_empty());
        assert_eq!(bridge.state.truncated_bursts.count(), 1);
        // Raw, those bytes would be the start of that OBU and the header
        // would be lost in it.
        bridge.burst(&HEADER);
        assert_eq!(bridge.state.descriptors, HEADER);
        assert_eq!(bridge.state.truncated_bursts.count(), 1);

        let mut raw = Harness::new();
        raw.raw(&[6 << 3, 100, 0xAA, 0xBB, 0xCC]);
        raw.raw(&HEADER);
        assert!(raw.state.descriptors.is_empty());
    }

    /// Before a sequence is open, a repeated copy of the descriptors opens
    /// it: that is how a receiver joins a stream already running.
    #[test]
    fn a_redundant_copy_opens_the_sequence_of_a_late_receiver() {
        let mut bridge = Harness::new();
        let mut copy = HEADER;
        copy[0] |= REDUNDANT;
        // Units of a sequence it never saw the start of: dropped.
        bridge.burst(&[6 << 3, 0x02, 0xAB, 0xCD]);
        assert!(bridge.state.descriptors.is_empty());
        bridge.burst(&copy);
        assert_eq!(bridge.state.descriptors, copy);
    }

    #[test]
    fn bursts_decode_as_the_raw_stream_does() {
        let Some(stream) = vector("test_000800.iamf") else {
            eprintln!("skipping: set HARLETTY_IAMF_VECTORS to the libiamf test vectors");
            return;
        };
        let mut raw = Harness::new();
        let expected = pcm(&raw.raw(&stream).frames);
        assert!(expected.len() > 100_000);

        for per_burst in [1, 2, 7] {
            let carried = Carried::of(&stream, per_burst);
            let mut bridge = Harness::new();
            let mut frames = bridge.bursts(std::slice::from_ref(&carried.descriptors));
            for (index, unit) in carried.units.iter().enumerate() {
                // A copy of the descriptors now and then, as a sender
                // repeats them: ignored while the sequence is open.
                if index % 5 == 4 {
                    let mut payload = carried.redundant_copy();
                    payload.extend_from_slice(unit);
                    frames.extend(bridge.bursts(&[payload]));
                } else {
                    frames.extend(bridge.bursts(std::slice::from_ref(unit)));
                }
            }
            assert_eq!(pcm(&frames), expected, "{per_burst} OBU(s) to a burst");
            assert_eq!(bridge.state.truncated_bursts.count(), 0);
        }
    }

    /// A receiver that joins after the start decodes from the first copy of
    /// the descriptors on, and what it decodes is the stream's.
    #[test]
    fn a_late_receiver_decodes_from_the_next_copy() {
        let Some(stream) = vector("test_000800.iamf") else {
            eprintln!("skipping: set HARLETTY_IAMF_VECTORS to the libiamf test vectors");
            return;
        };
        let carried = Carried::of(&stream, 1);
        let mut whole = Harness::new();
        let mut per_unit = vec![
            whole
                .bursts(std::slice::from_ref(&carried.descriptors))
                .len(),
        ];
        for unit in &carried.units {
            per_unit.push(whole.bursts(std::slice::from_ref(unit)).len());
        }

        let joined_at = carried.units.len() / 3;
        let mut late = Harness::new();
        // Missed: the descriptors and the first third of the units. Heard
        // without a sequence: a few units, which decode to nothing.
        let heard_blind = 3;
        let frames = late.bursts(&carried.units[joined_at..joined_at + heard_blind]);
        assert!(frames.is_empty());
        let mut payload = carried.redundant_copy();
        payload.extend_from_slice(&carried.units[joined_at + heard_blind]);
        let mut frames = late.bursts(&[payload]);
        frames.extend(late.bursts(&carried.units[joined_at + heard_blind + 1..]));

        // LPCM units stand alone: the tail is the stream's tail, sample for
        // sample.
        let mut reference = Harness::new();
        let all = reference.raw(&stream).frames;
        let skipped: usize = per_unit[..1 + joined_at + heard_blind].iter().sum();
        assert!(!frames.is_empty());
        assert_eq!(pcm(&frames), pcm(&all[skipped..]));
    }
}
