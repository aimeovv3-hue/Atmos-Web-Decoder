//! IAMF over the private IEC 61937 burst type decodes exactly as the raw
//! stream does (Omniphony `docs/iamf-transport-plan.md`): every conformance
//! vector the sender can carry is cut into bursts by Omniphony's framer,
//! read back by its parser the way a host does, and compared frame for
//! frame with the raw decode. A receiver that joins late is checked too.
//!
//! Needs `HARLETTY_IAMF_VECTORS` (the libiamf test vectors); skips without.
#![cfg(feature = "iamf")]

use abi_stable::std_types::RSlice;
use bridge_api::{RDecodedFrame, RInputTransport};
use spdif::SpdifParser;
use spdif::iamf::{Carrier, IamfSequence, write_bursts};

/// Wide enough for every vector that is not many channels of LPCM.
const CARRIER: Carrier = Carrier {
    rate_hz: 48_000,
    channels: 16,
};

fn vectors() -> Vec<std::path::PathBuf> {
    let Some(dir) = std::env::var_os("HARLETTY_IAMF_VECTORS") else {
        eprintln!("skipping: set HARLETTY_IAMF_VECTORS to the libiamf test vectors");
        return Vec::new();
    };
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "iamf"))
        .collect();
    paths.sort();
    paths
}

/// What a host reads of the frames: the samples, their shape, and the
/// object metadata.
fn heard(frames: &[RDecodedFrame]) -> Vec<String> {
    frames
        .iter()
        .map(|frame| {
            let metadata: Vec<String> = frame
                .metadata
                .iter()
                .map(|m| {
                    let names: Vec<(u32, &str)> = m
                        .name_updates
                        .iter()
                        .map(|update| (update.id, update.name.as_str()))
                        .collect();
                    format!(
                        "@{} ramp {} {:?} {:?} {:?} {names:?}",
                        m.sample_pos, m.ramp_duration, m.events, m.object_channels, m.channel_gains
                    )
                })
                .collect();
            format!(
                "{} Hz {} x {} {:?} {:?} {metadata:?}",
                frame.sampling_frequency,
                frame.sample_count,
                frame.channel_count,
                frame.channel_labels,
                frame.pcm,
            )
        })
        .collect()
}

/// The same OBUs on the raw transport, a push for the descriptors and for
/// each unit: `None` when the bridge refuses the sequence.
fn decode_raw(sequence: &IamfSequence) -> Option<Vec<RDecodedFrame>> {
    let mut bridge = harletty_bridge::new_bridge(false);
    let mut frames = Vec::new();
    let pushes = std::iter::once(&sequence.descriptors).chain(&sequence.units);
    for push in pushes {
        let result = bridge.push_packet(RSlice::from_slice(push), RInputTransport::Raw, 0);
        if !result.error_message.is_empty() {
            return None;
        }
        frames.extend(result.frames);
    }
    Some(frames)
}

/// Decode a carrier's bytes as a host does: its parser, then the bridge,
/// burst by burst.
fn decode_carried(name: &str, carried: &[u8]) -> Vec<RDecodedFrame> {
    let mut bridge = harletty_bridge::new_bridge(false);
    let mut parser = SpdifParser::new();
    let mut frames = Vec::new();
    // Odd-sized reads: bursts straddle them.
    for chunk in carried.chunks(4093) {
        parser.push_bytes(chunk);
        while let Some(burst) = parser.get_next_packet() {
            let result = bridge.push_packet(
                RSlice::from_slice(&burst.payload),
                RInputTransport::Iec61937,
                burst.data_type,
            );
            assert!(
                result.error_message.is_empty(),
                "{name}: {}",
                result.error_message
            );
            frames.extend(result.frames);
        }
    }
    frames
}

#[test]
fn every_carriable_vector_decodes_as_its_raw_stream() {
    let mut carried_count = 0;
    let mut frames_count = 0;
    for path in vectors() {
        let stream = std::fs::read(&path).unwrap();
        // What the sender cannot carry (several sequences, a unit wider
        // than the carrier, a rate that does not divide) is its to refuse.
        let Ok(sequence) = IamfSequence::parse(&stream) else {
            continue;
        };
        let Ok(period) = CARRIER.period_bytes(sequence.timing, Some(48_000)) else {
            continue;
        };
        let mut carried = Vec::new();
        if write_bursts(&mut carried, &sequence, period, 5).is_err() {
            continue;
        }
        // The reference is the same OBUs raw, unit by unit. A vector the
        // bridge refuses raw (a profile it does not decode) or that no
        // family claims (a wrong `ia_code`) has nothing to compare.
        let Some(raw) = decode_raw(&sequence) else {
            continue;
        };
        if raw.is_empty() {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy();
        let ours = decode_carried(&name, &carried);
        assert_eq!(ours.len(), raw.len(), "{name}: frames");
        assert_eq!(heard(&ours), heard(&raw), "{name}");
        carried_count += 1;
        frames_count += raw.len();
    }
    eprintln!("{carried_count} vector(s) carried, {frames_count} frame(s) compared");
    if std::env::var_os("HARLETTY_IAMF_VECTORS").is_some() {
        assert!(carried_count > 100, "{carried_count} vectors carried");
    }
}

/// A receiver that starts reading mid-stream, inside a burst, locks on the
/// next copy of the descriptors and decodes the stream's own samples.
#[test]
fn a_late_receiver_locks_on_the_next_copy() {
    let Some(dir) = std::env::var_os("HARLETTY_IAMF_VECTORS") else {
        eprintln!("skipping: set HARLETTY_IAMF_VECTORS to the libiamf test vectors");
        return;
    };
    // LPCM: its units stand alone, so the tail is comparable sample for
    // sample.
    let stream = std::fs::read(std::path::Path::new(&dir).join("test_000800.iamf")).unwrap();
    let sequence = IamfSequence::parse(&stream).unwrap();
    let period = CARRIER.period_bytes(sequence.timing, None).unwrap();
    let copy_every = 10;
    let mut carried = Vec::new();
    write_bursts(&mut carried, &sequence, period, copy_every).unwrap();

    let raw = decode_raw(&sequence).unwrap();
    let pcm = |frames: &[RDecodedFrame]| -> Vec<i32> {
        frames.iter().flat_map(|f| f.pcm.iter().copied()).collect()
    };
    let whole = pcm(&raw);
    // Joined 23 bursts and a bit in: the bursts up to the next copy (the
    // 30th) are heard without a sequence.
    let late = pcm(&decode_carried("late", &carried[23 * period + 1234..]));
    assert!(!late.is_empty());
    assert!(late.len() < whole.len());
    assert_eq!(late, whole[whole.len() - late.len()..]);
    let samples_per_unit = whole.len() / sequence.units.len();
    assert_eq!(
        (whole.len() - late.len()) / samples_per_unit,
        30,
        "locked at the first copy after joining"
    );
}

/// A damaged burst costs its own unit and no other. In a unit of several
/// substreams, the OBUs of a truncated burst that still frame must not be
/// decoded: their substreams would wait in the decoder for the rest of
/// their unit and be matched with the next one's, shifting every unit after.
#[test]
fn a_damaged_burst_costs_its_unit_only() {
    let Some(dir) = std::env::var_os("HARLETTY_IAMF_VECTORS") else {
        eprintln!("skipping: set HARLETTY_IAMF_VECTORS to the libiamf test vectors");
        return;
    };
    // LPCM 5.1: four substreams to a unit.
    let stream = std::fs::read(std::path::Path::new(&dir).join("test_000038.iamf")).unwrap();
    let sequence = IamfSequence::parse(&stream).unwrap();
    assert!(sequence.units.len() > 12);

    // How unit 5 is damaged: its last byte lost (the last OBU no longer
    // frames, the ones before it still do), or its last OBU lost whole (what
    // is left is whole OBUs, of an incomplete unit).
    let last_obu = |unit: &[u8]| -> usize {
        let (mut pos, mut last) = (0, 0);
        while let Ok(Some(obu)) = bridge_family_iamf::frame_obu(&unit[pos..]) {
            last = pos;
            pos += obu.len;
        }
        last
    };
    let decode = |damaged: Option<usize>, whole_obu: bool| -> Vec<Vec<i32>> {
        let mut bridge = harletty_bridge::new_bridge(false);
        let mut units = Vec::new();
        for (index, unit) in sequence.units.iter().enumerate() {
            let mut payload = if index == 0 {
                sequence.descriptors.clone()
            } else {
                Vec::new()
            };
            payload.extend_from_slice(unit);
            if damaged == Some(index) {
                let keep = if whole_obu {
                    payload.len() - (unit.len() - last_obu(unit))
                } else {
                    payload.len() - 1
                };
                payload.truncate(keep);
            }
            let result = bridge.push_packet(
                RSlice::from_slice(&payload),
                RInputTransport::Iec61937,
                spdif::DATA_TYPE_IAMF_PRIVATE,
            );
            assert!(result.error_message.is_empty(), "{}", result.error_message);
            units.push(
                result
                    .frames
                    .iter()
                    .flat_map(|f| f.pcm.iter().copied())
                    .collect(),
            );
        }
        units
    };

    let intact = decode(None, false);
    assert!(intact.iter().all(|unit| !unit.is_empty()));
    for whole_obu in [false, true] {
        let hurt = decode(Some(5), whole_obu);
        assert!(hurt[5].is_empty(), "the damaged unit is not heard");
        for (index, (ours, reference)) in hurt.iter().zip(&intact).enumerate() {
            if index != 5 {
                assert_eq!(ours, reference, "unit {index}, whole OBU lost: {whole_obu}");
            }
        }
    }
}
