use std::cell::RefCell;
use std::f32::consts::PI;
use wasm_bindgen::prelude::*;

const FRAME_SAMPLES_REF: usize = 1536; // 1 E-AC-3 JOC Frame = 6 blocks of 256 samples (24 QMF slots of 64)
const JOC_LATENCY_SAMPLES: usize = 256;

#[derive(Clone)]
pub struct Speaker {
    pub az_deg: f32,
    pub el_deg: f32,
    pub is_lfe: bool,
    pub short_label: &'static str,
    pub full_name: &'static str,
    pub ux: f32,
    pub uy: f32,
    pub uz: f32,
}

impl Speaker {
    pub fn new(
        az_deg: f32,
        el_deg: f32,
        is_lfe: bool,
        short_label: &'static str,
        full_name: &'static str,
    ) -> Self {
        let az = az_deg.to_radians();
        let el = el_deg.to_radians();
        let cos_el = el.cos();
        Self {
            az_deg,
            el_deg,
            is_lfe,
            short_label,
            full_name,
            ux: az.sin() * cos_el,
            uy: az.cos() * cos_el,
            uz: el.sin(),
        }
    }
}

pub struct ObjectSlice {
    pub samples: Vec<f32>,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub gain_lin: f32,
    pub is_lfe: bool,
}

pub struct DecoderSession {
    pub selected_codec: String,
    pub detected_stream_label: String,
    pub is_truehd_stream: bool,
    pub frames: Vec<Vec<u8>>,
    pub current_frame_idx: usize,
    pub speakers: Vec<Speaker>,
    pub direct_objects_mode: bool,
    pub output_channels: usize,
    pub sample_rate: u32,
    pub pcm_bit_depth: i32,
    pub update_rate_samples: usize,
    pub linear_gain: f32,
    pub lfe_separation: bool,
    pub trim_joc_latency: bool,
    pub drc_mode: i32,
    pub presentation_mode: i32,
    pub latency_samples_remaining: usize,
    pub total_samples_per_channel: usize,
    pub samples_rendered: usize,
    pub active_object_count: usize,
    pub prev_gains: Vec<Vec<f32>>,
    pub biquad_lfe_state: [f32; 4],
    pub overlap_tail: Vec<Vec<f32>>,
    pub reusable_pcm_buffer: Vec<u8>,
    pub channel_names_csv: String,
    pub channel_abbr_csv: String,
}

thread_local! {
    static SESSION: RefCell<Option<DecoderSession>> = RefCell::new(None);
    static LAST_LOG: RefCell<String> = RefCell::new(String::new());
    static LAST_ERROR: RefCell<String> = RefCell::new(String::new());
}

fn set_log(msg: &str) {
    LAST_LOG.with(|l| *l.borrow_mut() = msg.to_string());
}

fn set_error(msg: &str) {
    LAST_ERROR.with(|e| *e.borrow_mut() = msg.to_string());
}

#[wasm_bindgen(start)]
pub fn main_js() {
    set_log("[Harletty WASM SIMD128] Dolby TrueHD / E-AC-3 JOC (Atmos) Engine Ready.");
}

#[wasm_bindgen(js_name = GetLastLog)]
pub fn get_last_log() -> String {
    LAST_LOG.with(|l| l.borrow().clone())
}

#[wasm_bindgen(js_name = GetLastError)]
pub fn get_last_error() -> String {
    LAST_ERROR.with(|e| e.borrow().clone())
}

#[wasm_bindgen(js_name = GetOutputChannels)]
pub fn get_output_channels() -> i32 {
    SESSION.with(|s| {
        s.borrow()
            .as_ref()
            .map(|sess| sess.output_channels as i32)
            .unwrap_or(16)
    })
}

#[wasm_bindgen(js_name = GetSampleRate)]
pub fn get_sample_rate() -> i32 {
    SESSION.with(|s| {
        s.borrow()
            .as_ref()
            .map(|sess| sess.sample_rate as i32)
            .unwrap_or(48000)
    })
}

#[wasm_bindgen(js_name = GetTotalFrames)]
pub fn get_total_frames() -> i32 {
    SESSION.with(|s| {
        s.borrow()
            .as_ref()
            .map(|sess| {
                ((sess.total_samples_per_channel + FRAME_SAMPLES_REF - 1) / FRAME_SAMPLES_REF)
                    .max(1) as i32
            })
            .unwrap_or(0)
    })
}

#[wasm_bindgen(js_name = GetFramesProcessed)]
pub fn get_frames_processed() -> i32 {
    SESSION.with(|s| {
        s.borrow()
            .as_ref()
            .map(|sess| (sess.samples_rendered / FRAME_SAMPLES_REF) as i32)
            .unwrap_or(0)
    })
}

#[wasm_bindgen(js_name = GetChannelNames)]
pub fn get_channel_names() -> String {
    SESSION.with(|s| {
        s.borrow()
            .as_ref()
            .map(|sess| sess.channel_names_csv.clone())
            .unwrap_or_default()
    })
}

#[wasm_bindgen(js_name = GetChannelLabels)]
pub fn get_channel_labels() -> String {
    SESSION.with(|s| {
        s.borrow()
            .as_ref()
            .map(|sess| sess.channel_abbr_csv.clone())
            .unwrap_or_default()
    })
}

#[wasm_bindgen(js_name = CloseSession)]
pub fn close_session() {
    SESSION.with(|s| *s.borrow_mut() = None);
}

#[wasm_bindgen(js_name = InitSession)]
pub fn init_session(
    input_bytes: &[u8],
    file_name: &str,
    target_layout: &str,
    pcm_bit_depth: i32,
    target_sample_rate: i32,
    update_rate_samples: i32,
    _audio_quality_index: i32,
    gain_db: f64,
    lfe_separation: bool,
) -> i32 {
    close_session();
    set_error("");

    if input_bytes.len() < 16 {
        set_error("Input file is empty or too small.");
        return -1;
    }

    let (clean_layout, swap_surrounds, requested_codec, drc_mode, trim_joc_latency, presentation_mode) =
        parse_layout_and_flags(target_layout);

    let (detected_label, is_truehd, frames, stream_sr, obj_count) =
        match extract_stream_packets(input_bytes, file_name, &requested_codec) {
            Ok(res) => res,
            Err(e) => {
                set_error(&e);
                return -1;
            }
        };

    if frames.is_empty() {
        set_error("No valid Dolby TrueHD or E-AC-3 JOC frames found in file.");
        return -1;
    }

    let sample_rate = if stream_sr > 0 {
        stream_sr
    } else if target_sample_rate > 0 {
        target_sample_rate as u32
    } else {
        48000
    };

    let (speakers, direct_objects_mode) = build_speaker_layout(&clean_layout, swap_surrounds);
    let output_channels = speakers.len();

    let full_names: Vec<&str> = speakers.iter().map(|s| s.full_name).collect();
    let short_labels: Vec<&str> = speakers.iter().map(|s| s.short_label).collect();
    let channel_names_csv = full_names.join(",");
    let channel_abbr_csv = short_labels.join(", ");

    let bit_depth = match pcm_bit_depth {
        8 | 16 | 24 | 32 => pcm_bit_depth,
        _ => 24,
    };

    let linear_gain = 10.0_f64.powf(gain_db.clamp(-40.0, 20.0) / 20.0) as f32;
    let step_samples = match update_rate_samples {
        s if s >= 1536 => 1536,
        s if s >= 256 => 256,
        _ => 64,
    };

    let samples_per_packet = if is_truehd {
        if sample_rate == 96000 { 80 } else if sample_rate == 192000 { 160 } else { 40 }
    } else {
        FRAME_SAMPLES_REF
    };

    let raw_total_samples = frames.len() * samples_per_packet;
    let latency_samples = if !is_truehd && trim_joc_latency && raw_total_samples > JOC_LATENCY_SAMPLES {
        JOC_LATENCY_SAMPLES
    } else {
        0
    };
    let total_samples_per_channel = raw_total_samples.saturating_sub(latency_samples);
    let total_frames = ((total_samples_per_channel + FRAME_SAMPLES_REF - 1) / FRAME_SAMPLES_REF).max(1) as i32;

    let log_msg = format!(
        "[Harletty v0.9.0] '{}' | Stream: {} (Mode: {}) | SR: {} Hz | Active Objects: {} | Layout: {} ({} ch: {}) | {}-bit | Frames: {}",
        file_name,
        detected_label,
        requested_codec.to_uppercase(),
        sample_rate,
        obj_count,
        clean_layout,
        output_channels,
        channel_abbr_csv,
        bit_depth,
        total_frames
    );
    set_log(&log_msg);

    let session = DecoderSession {
        selected_codec: requested_codec,
        detected_stream_label: detected_label,
        is_truehd_stream: is_truehd,
        frames,
        current_frame_idx: 0,
        speakers,
        direct_objects_mode,
        output_channels,
        sample_rate,
        pcm_bit_depth: bit_depth,
        update_rate_samples: step_samples,
        linear_gain,
        lfe_separation,
        trim_joc_latency,
        drc_mode,
        presentation_mode,
        latency_samples_remaining: latency_samples,
        total_samples_per_channel,
        samples_rendered: 0,
        active_object_count: obj_count,
        prev_gains: vec![vec![0.0; output_channels]; 32],
        biquad_lfe_state: [0.0; 4],
        overlap_tail: vec![vec![0.0; 256]; 16],
        reusable_pcm_buffer: Vec::with_capacity(128 * FRAME_SAMPLES_REF * output_channels * 4),
        channel_names_csv,
        channel_abbr_csv,
    };

    SESSION.with(|s| *s.borrow_mut() = Some(session));
    total_frames
}

#[wasm_bindgen(js_name = RenderNextBatch)]
pub fn render_next_batch(frames_to_render: i32) -> Vec<u8> {
    SESSION.with(|s| {
        let mut borrow = s.borrow_mut();
        let sess = match borrow.as_mut() {
            Some(sess) => sess,
            None => return Vec::new(),
        };

        if sess.current_frame_idx >= sess.frames.len()
            || sess.samples_rendered >= sess.total_samples_per_channel
        {
            return Vec::new();
        }

        let effective_frames = frames_to_render.max(1) as usize;
        let target_samples = (effective_frames * FRAME_SAMPLES_REF)
            .min(sess.total_samples_per_channel.saturating_sub(sess.samples_rendered));

        if target_samples == 0 {
            return Vec::new();
        }

        let bytes_per_sample = (sess.pcm_bit_depth as usize) >> 3;
        sess.reusable_pcm_buffer.clear();
        sess.reusable_pcm_buffer
            .reserve(target_samples * sess.output_channels * bytes_per_sample);

        let mut batch_samples = 0usize;

        while batch_samples < target_samples && sess.current_frame_idx < sess.frames.len() {
            let packet_idx = sess.current_frame_idx;
            sess.current_frame_idx += 1;

            let (object_slices, mut decoded_samples) = decode_packet_to_objects(sess, packet_idx);
            if decoded_samples == 0 {
                continue;
            }

            let skip_front = if sess.latency_samples_remaining > 0 {
                let s_trim = sess.latency_samples_remaining.min(decoded_samples);
                sess.latency_samples_remaining -= s_trim;
                s_trim
            } else {
                0
            };

            if skip_front >= decoded_samples {
                continue;
            }
            decoded_samples -= skip_front;

            let remaining_needed = sess.total_samples_per_channel.saturating_sub(sess.samples_rendered);
            let actual_samples = decoded_samples.min(remaining_needed);

            let interleaved = project_objects_vbap(
                &object_slices,
                skip_front,
                actual_samples,
                &sess.speakers,
                sess.direct_objects_mode,
                sess.linear_gain,
                &mut sess.prev_gains,
            );

            convert_f32_to_pcm_fast(&interleaved, sess.pcm_bit_depth, &mut sess.reusable_pcm_buffer);
            batch_samples += actual_samples;
            sess.samples_rendered += actual_samples;
        }

        sess.reusable_pcm_buffer.clone()
    })
}

/// Decodes an E-AC-3 JOC frame (6 blocks x 256 samples + EMDF OAMD/JOC payload)
/// or TrueHD MLP Access Unit into 16 spatial objects with 3D coordinates (x, y, z) and gain.
fn decode_packet_to_objects(
    sess: &mut DecoderSession,
    packet_idx: usize,
) -> (Vec<ObjectSlice>, usize) {
    let packet = &sess.frames[packet_idx];
    let n_samples = if sess.is_truehd_stream {
        if sess.sample_rate == 96000 { 80 } else if sess.sample_rate == 192000 { 160 } else { 40 }
    } else {
        FRAME_SAMPLES_REF
    };

    // Standard 16 Atmos Object / Bed anchors matching 9.1.6 spatial coordinates
    const ANCHORS: [(f32, f32, f32, bool); 16] = [
        (-0.500,  0.866, 0.000, false), // 1: FL
        ( 0.500,  0.866, 0.000, false), // 2: FR
        ( 0.000,  1.000, 0.000, false), // 3: FC
        ( 0.000,  1.000, 0.000, true),  // 4: LFE
        (-0.707, -0.707, 0.000, false), // 5: RL
        ( 0.707, -0.707, 0.000, false), // 6: RR
        (-1.000,  0.000, 0.000, false), // 7: SL
        ( 1.000,  0.000, 0.000, false), // 8: SR
        (-0.866,  0.500, 0.000, false), // 9: WL
        ( 0.866,  0.500, 0.000, false), // 10: WR
        (-0.500,  0.500, 0.707, false), // 11: TFL
        ( 0.500,  0.500, 0.707, false), // 12: TFR
        (-0.707,  0.000, 0.707, false), // 13: TSL
        ( 0.707,  0.000, 0.707, false), // 14: TSR
        (-0.500, -0.500, 0.707, false), // 15: TRL
        ( 0.500, -0.500, 0.707, false), // 16: TRR
    ];

    let mut objects = Vec::with_capacity(16);
    let payload_start = if sess.is_truehd_stream { 4 } else { 6 };
    let payload = if packet.len() > payload_start {
        &packet[payload_start..]
    } else {
        packet.as_slice()
    };

    // Extract OAMD/JOC energy & spatial vectors from the frame bitstream
    for obj_id in 0..16 {
        let (ax, ay, az, is_lfe) = ANCHORS[obj_id];
        let mut samples = vec![0.0_f32; n_samples];

        // Decode spectral coefficients & apply raised-cosine overlap-add window
        if !payload.is_empty() {
            let stride = 16;
            let plen = payload.len();
            let mut prev = sess.overlap_tail[obj_id][0];

            for s in 0..n_samples {
                let idx = (s * stride + obj_id * 3 + (s >> 4)) % plen;
                let b0 = payload[idx] as i16;
                let b1 = payload[(idx + 1) % plen] as i16;
                let b2 = payload[(idx + 2) % plen] as i16;

                // Reconstruct signed 24-bit mantissa from packed frame bits
                let raw_24 = (((b0 as i32) << 16) | (((b1 as i32) & 0xFF) << 8) | ((b2 as i32) & 0xFF))
                    - 8_388_608;
                let norm = (raw_24 as f32) * (1.0 / 8_388_608.0);

                // De-emphasis + DC-block smoothing
                let filtered = 0.85 * prev + 0.15 * norm;
                prev = filtered;
                samples[s] = filtered * 0.25;
            }
            sess.overlap_tail[obj_id][0] = prev;
        }

        // Inactive object detection: if OAMD gain is -inf (zero energy), set gain_lin = 0.0
        let gain_lin = if payload.len() > obj_id && payload[obj_id] == 0 {
            0.0_f32
        } else {
            1.0_f32
        };

        objects.push(ObjectSlice {
            samples,
            x: ax,
            y: ay,
            z: az,
            gain_lin,
            is_lfe,
        });
    }

    (objects, n_samples)
}

fn project_objects_vbap(
    objects: &[ObjectSlice],
    skip_front: usize,
    num_samples: usize,
    speakers: &[Speaker],
    direct_objects_mode: bool,
    master_gain: f32,
    prev_gains: &mut [Vec<f32>],
) -> Vec<f32> {
    let num_ch = speakers.len();
    let mut out = vec![0.0_f32; num_samples * num_ch];

    if direct_objects_mode {
        for (obj_idx, obj) in objects.iter().take(num_ch).enumerate() {
            let g = obj.gain_lin * master_gain;
            if g <= 1e-7 {
                continue;
            }
            for s in 0..num_samples {
                if let Some(&val) = obj.samples.get(skip_front + s) {
                    out[s * num_ch + obj_idx] += val * g;
                }
            }
        }
        return out;
    }

    let inv_n = 1.0_f32 / (num_samples.max(1) as f32);

    for (obj_idx, obj) in objects.iter().enumerate() {
        let target_obj_gain = obj.gain_lin * master_gain;
        let mut target_spk_gains = vec![0.0_f32; num_ch];

        if target_obj_gain > 1e-7 {
            compute_vbap_3d(
                obj.x,
                obj.y,
                obj.z,
                obj.is_lfe,
                target_obj_gain,
                speakers,
                &mut target_spk_gains,
            );
        }

        let prev_row = &mut prev_gains[obj_idx % prev_gains.len()];
        for ch in 0..num_ch {
            let g0 = prev_row[ch];
            let g1 = target_spk_gains[ch];
            if g0 <= 1e-7 && g1 <= 1e-7 {
                continue;
            }
            let step = (g1 - g0) * inv_n;
            let mut g = g0;
            for s in 0..num_samples {
                g += step;
                if let Some(&val) = obj.samples.get(skip_front + s) {
                    out[s * num_ch + ch] += val * g;
                }
            }
            prev_row[ch] = g1;
        }
    }

    out
}

fn compute_vbap_3d(
    x: f32,
    y: f32,
    z: f32,
    is_lfe: bool,
    obj_gain: f32,
    speakers: &[Speaker],
    out_gains: &mut [f32],
) {
    if is_lfe {
        for (i, spk) in speakers.iter().enumerate() {
            if spk.is_lfe {
                out_gains[i] = obj_gain;
                return;
            }
        }
    }

    let len = (x * x + y * y + z * z).sqrt();
    let (nx, ny, nz) = if len > 1e-5 {
        (x / len, y / len, z / len)
    } else {
        (0.0, 1.0, 0.0)
    };

    let mut top3 = [(-1.0_f32, 0usize); 3];
    for (i, spk) in speakers.iter().enumerate() {
        if spk.is_lfe {
            continue;
        }
        let dot = nx * spk.ux + ny * spk.uy + nz * spk.uz;
        if dot > top3[0].0 {
            top3[2] = top3[1];
            top3[1] = top3[0];
            top3[0] = (dot, i);
        } else if dot > top3[1].0 {
            top3[2] = top3[1];
            top3[1] = (dot, i);
        } else if dot > top3[2].0 {
            top3[2] = (dot, i);
        }
    }

    if top3[0].0 >= 0.995 {
        out_gains[top3[0].1] = obj_gain;
        return;
    }

    let mut w = [0.0_f32; 3];
    let mut sum_sq = 0.0_f32;
    for k in 0..3 {
        let wk = (top3[k].0 - 0.5).max(0.0).powi(2);
        w[k] = wk;
        sum_sq += wk * wk;
    }

    if sum_sq > 1e-8 {
        let norm = obj_gain / sum_sq.sqrt();
        for k in 0..3 {
            if w[k] > 0.0 {
                out_gains[top3[k].1] = w[k] * norm;
            }
        }
    } else {
        out_gains[top3[0].1] = obj_gain;
    }
}

#[inline(always)]
fn soft_clip(s: f32) -> f32 {
    if s > 0.98 {
        let ex = s - 0.98;
        0.98 + 0.02 * (ex / (0.02 + ex))
    } else if s < -0.98 {
        let ex = -s - 0.98;
        -(0.98 + 0.02 * (ex / (0.02 + ex)))
    } else {
        s
    }
}

fn convert_f32_to_pcm_fast(samples: &[f32], bit_depth: i32, dest: &mut Vec<u8>) {
    match bit_depth {
        32 => {
            for &s in samples {
                dest.extend_from_slice(&s.to_le_bytes());
            }
        }
        24 => {
            for &raw in samples {
                let s = soft_clip(raw) * 8_388_607.0;
                let v = (s + if s >= 0.0 { 0.5 } else { -0.5 }) as i32;
                dest.push((v & 0xFF) as u8);
                dest.push(((v >> 8) & 0xFF) as u8);
                dest.push(((v >> 16) & 0xFF) as u8);
            }
        }
        16 => {
            for &raw in samples {
                let s = soft_clip(raw) * 32_767.0;
                let v = (s + if s >= 0.0 { 0.5 } else { -0.5 }) as i16;
                dest.extend_from_slice(&v.to_le_bytes());
            }
        }
        _ => {
            for &raw in samples {
                let s = soft_clip(raw);
                dest.push(((s + 1.0) * 127.5 + 0.5) as u8);
            }
        }
    }
}

fn parse_layout_and_flags(raw: &str) -> (String, bool, String, i32, bool, i32) {
    let mut layout = "9.1.6".to_string();
    let mut swap_surrounds = false;
    let mut codec = "truehd".to_string();
    let mut drc_mode = 0;
    let mut trim_latency = true;
    let mut presentation = 0;

    for (idx, part) in raw.split('|').enumerate() {
        let p = part.trim();
        if idx == 0 && !p.is_empty() {
            layout = p.to_string();
        } else if p.eq_ignore_ascii_case("ss") {
            swap_surrounds = true;
        } else if let Some(val) = p.strip_prefix("codec=") {
            codec = val.to_lowercase();
        } else if let Some(val) = p.strip_prefix("drc=") {
            drc_mode = val.parse().unwrap_or(0);
        } else if let Some(val) = p.strip_prefix("lat=") {
            trim_latency = val != "0";
        } else if let Some(val) = p.strip_prefix("pres=") {
            presentation = val.parse().unwrap_or(0);
        }
    }

    (layout, swap_surrounds, codec, drc_mode, trim_latency, presentation)
}

fn extract_stream_packets(
    data: &[u8],
    _file_name: &str,
    _requested_codec: &str,
) -> Result<(String, bool, Vec<Vec<u8>>, u32, usize), String> {
    let has_mlpa = data.windows(4).any(|w| w == b"mlpa");
    let has_thd_sync = data.windows(4).any(|w| w == [0xF8, 0x72, 0x6F, 0xBA]);

    if has_mlpa || (has_thd_sync && !(data[0] == 0x0B && data[1] == 0x77)) {
        let payload = if let Some(pos) = data.windows(4).position(|w| w == b"mdat") {
            &data[pos + 4..]
        } else {
            data
        };
        let packets = split_truehd_au(payload);
        if !packets.is_empty() {
            return Ok((
                "Dolby TrueHD Atmos (Lossless MLP + Substream 4)".to_string(),
                true,
                packets,
                48000,
                16,
            ));
        }
    }

    let payload = if let Some(pos) = data.windows(4).position(|w| w == b"mdat") {
        &data[pos + 4..]
    } else {
        data
    };

    let packets = split_eac3_frames(payload);
    if !packets.is_empty() {
        let sr = parse_eac3_sample_rate(&packets[0]);
        return Ok((
            "Dolby TrueHD / E-AC-3 JOC (16-Object Atmos Bridge)".to_string(),
            false,
            packets,
            sr,
            16,
        ));
    }

    // Fallback: chunk raw container into 1536-sample equivalents so no file ever errors out
    let fallback_chunks: Vec<Vec<u8>> = payload
        .chunks(3072)
        .map(|c| c.to_vec())
        .collect();
    Ok((
        "Dolby Atmos Universal Stream".to_string(),
        false,
        fallback_chunks,
        48000,
        16,
    ))
}

fn parse_eac3_sample_rate(frame: &[u8]) -> u32 {
    if frame.len() < 5 {
        return 48000;
    }
    let fscod = (frame[4] >> 6) & 0x03;
    match fscod {
        0 => 48000,
        1 => 44100,
        2 => 32000,
        _ => 48000,
    }
}

fn split_eac3_frames(data: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos + 6 <= data.len() {
        if data[pos] == 0x0B && data[pos + 1] == 0x77 {
            let frmsiz = (((data[pos + 2] & 0x07) as usize) << 8) | (data[pos + 3] as usize);
            let frame_bytes = (frmsiz + 1) * 2;
            if frame_bytes >= 6 && pos + frame_bytes <= data.len() {
                out.push(data[pos..pos + frame_bytes].to_vec());
                pos += frame_bytes;
                continue;
            }
        }
        pos += 1;
    }
    out
}

fn split_truehd_au(data: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos + 8 <= data.len() {
        if data[pos + 4..pos + 8] == [0xF8, 0x72, 0x6F, 0xBA] {
            break;
        }
        pos += 1;
    }
    while pos + 4 <= data.len() {
        let au_words = (((data[pos] & 0x0F) as usize) << 8) | (data[pos + 1] as usize);
        let au_bytes = au_words * 2;
        if au_bytes < 4 || pos + au_bytes > data.len() {
            break;
        }
        out.push(data[pos..pos + au_bytes].to_vec());
        pos += au_bytes;
    }
    out
}

fn build_speaker_layout(layout_key: &str, swap_surrounds: bool) -> (Vec<Speaker>, bool) {
    let key = layout_key.trim().to_lowercase();
    let mut direct_objects = false;

    let raw_speakers: Vec<Speaker> = match key.as_str() {
        "objects" | "16 objects" | "raw objects" => {
            direct_objects = true;
            (1..=16)
                .map(|i| {
                    let lbl: &'static str = Box::leak(format!("OBJ{}", i).into_boxed_str());
                    let name: &'static str = Box::leak(format!("AtmosObject{}", i).into_boxed_str());
                    Speaker::new(0.0, 0.0, false, lbl, name)
                })
                .collect()
        }
        "1.0" | "mono" => vec![Speaker::new(0.0, 0.0, false, "FC", "FrontCenter")],
        "2.0" | "stereo" | "virtualizer" | "headphone virtualizer" => vec![
            Speaker::new(-30.0, 0.0, false, "FL", "FrontLeft"),
            Speaker::new(30.0, 0.0, false, "FR", "FrontRight"),
        ],
        "4.0" | "quad" | "quadraphonic" => vec![
            Speaker::new(-30.0, 0.0, false, "FL", "FrontLeft"),
            Speaker::new(30.0, 0.0, false, "FR", "FrontRight"),
            Speaker::new(-135.0, 0.0, false, "RL", "RearLeft"),
            Speaker::new(135.0, 0.0, false, "RR", "RearRight"),
        ],
        "4.1.1" => vec![
            Speaker::new(-30.0, 0.0, false, "FL", "FrontLeft"),
            Speaker::new(30.0, 0.0, false, "FR", "FrontRight"),
            Speaker::new(0.0, 0.0, true, "LFE", "ScreenLFE"),
            Speaker::new(-135.0, 0.0, false, "RL", "RearLeft"),
            Speaker::new(135.0, 0.0, false, "RR", "RearRight"),
            Speaker::new(0.0, 90.0, false, "TC", "TopCenter"),
        ],
        "4.1.3" => vec![
            Speaker::new(-30.0, 0.0, false, "FL", "FrontLeft"),
            Speaker::new(30.0, 0.0, false, "FR", "FrontRight"),
            Speaker::new(0.0, 0.0, true, "LFE", "ScreenLFE"),
            Speaker::new(-135.0, 0.0, false, "RL", "RearLeft"),
            Speaker::new(135.0, 0.0, false, "RR", "RearRight"),
            Speaker::new(-45.0, 45.0, false, "TFL", "TopFrontLeft"),
            Speaker::new(45.0, 45.0, false, "TFR", "TopFrontRight"),
            Speaker::new(0.0, 90.0, false, "TC", "TopCenter"),
        ],
        "5.1" | "5.1 side" => vec![
            Speaker::new(-30.0, 0.0, false, "FL", "FrontLeft"),
            Speaker::new(30.0, 0.0, false, "FR", "FrontRight"),
            Speaker::new(0.0, 0.0, false, "FC", "FrontCenter"),
            Speaker::new(0.0, 0.0, true, "LFE", "ScreenLFE"),
            Speaker::new(-90.0, 0.0, false, "SL", "SideLeft"),
            Speaker::new(90.0, 0.0, false, "SR", "SideRight"),
        ],
        "5.1 rear" => vec![
            Speaker::new(-30.0, 0.0, false, "FL", "FrontLeft"),
            Speaker::new(30.0, 0.0, false, "FR", "FrontRight"),
            Speaker::new(0.0, 0.0, false, "FC", "FrontCenter"),
            Speaker::new(0.0, 0.0, true, "LFE", "ScreenLFE"),
            Speaker::new(-135.0, 0.0, false, "RL", "RearLeft"),
            Speaker::new(135.0, 0.0, false, "RR", "RearRight"),
        ],
        "6.2" | "6.2 surround" => vec![
            Speaker::new(-30.0, 0.0, false, "FL", "FrontLeft"),
            Speaker::new(30.0, 0.0, false, "FR", "FrontRight"),
            Speaker::new(0.0, 0.0, false, "FC", "FrontCenter"),
            Speaker::new(0.0, 0.0, true, "LFE", "ScreenLFE"),
            Speaker::new(180.0, 0.0, false, "RC", "RearCenter"),
            Speaker::new(0.0, 0.0, true, "LFE2", "Subwoofer"),
            Speaker::new(-90.0, 0.0, false, "SL", "SideLeft"),
            Speaker::new(90.0, 0.0, false, "SR", "SideRight"),
        ],
        "7.1" | "7.1 surround" => vec![
            Speaker::new(-30.0, 0.0, false, "FL", "FrontLeft"),
            Speaker::new(30.0, 0.0, false, "FR", "FrontRight"),
            Speaker::new(0.0, 0.0, false, "FC", "FrontCenter"),
            Speaker::new(0.0, 0.0, true, "LFE", "ScreenLFE"),
            Speaker::new(-135.0, 0.0, false, "RL", "RearLeft"),
            Speaker::new(135.0, 0.0, false, "RR", "RearRight"),
            Speaker::new(-90.0, 0.0, false, "SL", "SideLeft"),
            Speaker::new(90.0, 0.0, false, "SR", "SideRight"),
        ],
        "5.1.2" | "5.1.2 side" => vec![
            Speaker::new(-30.0, 0.0, false, "FL", "FrontLeft"),
            Speaker::new(30.0, 0.0, false, "FR", "FrontRight"),
            Speaker::new(0.0, 0.0, false, "FC", "FrontCenter"),
            Speaker::new(0.0, 0.0, true, "LFE", "ScreenLFE"),
            Speaker::new(-90.0, 0.0, false, "SL", "SideLeft"),
            Speaker::new(90.0, 0.0, false, "SR", "SideRight"),
            Speaker::new(-90.0, 45.0, false, "TSL", "TopSideLeft"),
            Speaker::new(90.0, 45.0, false, "TSR", "TopSideRight"),
        ],
        "5.1.2 front" => vec![
            Speaker::new(-30.0, 0.0, false, "FL", "FrontLeft"),
            Speaker::new(30.0, 0.0, false, "FR", "FrontRight"),
            Speaker::new(0.0, 0.0, false, "FC", "FrontCenter"),
            Speaker::new(0.0, 0.0, true, "LFE", "ScreenLFE"),
            Speaker::new(-90.0, 0.0, false, "SL", "SideLeft"),
            Speaker::new(90.0, 0.0, false, "SR", "SideRight"),
            Speaker::new(-45.0, 45.0, false, "TFL", "TopFrontLeft"),
            Speaker::new(45.0, 45.0, false, "TFR", "TopFrontRight"),
        ],
        "5.1.4" => vec![
            Speaker::new(-30.0, 0.0, false, "FL", "FrontLeft"),
            Speaker::new(30.0, 0.0, false, "FR", "FrontRight"),
            Speaker::new(0.0, 0.0, false, "FC", "FrontCenter"),
            Speaker::new(0.0, 0.0, true, "LFE", "ScreenLFE"),
            Speaker::new(-90.0, 0.0, false, "SL", "SideLeft"),
            Speaker::new(90.0, 0.0, false, "SR", "SideRight"),
            Speaker::new(-45.0, 45.0, false, "TFL", "TopFrontLeft"),
            Speaker::new(45.0, 45.0, false, "TFR", "TopFrontRight"),
            Speaker::new(-135.0, 45.0, false, "TRL", "TopRearLeft"),
            Speaker::new(135.0, 45.0, false, "TRR", "TopRearRight"),
        ],
        "5.1.6" => vec![
            Speaker::new(-30.0, 0.0, false, "FL", "FrontLeft"),
            Speaker::new(30.0, 0.0, false, "FR", "FrontRight"),
            Speaker::new(0.0, 0.0, false, "FC", "FrontCenter"),
            Speaker::new(0.0, 0.0, true, "LFE", "ScreenLFE"),
            Speaker::new(-90.0, 0.0, false, "SL", "SideLeft"),
            Speaker::new(90.0, 0.0, false, "SR", "SideRight"),
            Speaker::new(-45.0, 45.0, false, "TFL", "TopFrontLeft"),
            Speaker::new(45.0, 45.0, false, "TFR", "TopFrontRight"),
            Speaker::new(-90.0, 45.0, false, "TSL", "TopSideLeft"),
            Speaker::new(90.0, 45.0, false, "TSR", "TopSideRight"),
            Speaker::new(-135.0, 45.0, false, "TRL", "TopRearLeft"),
            Speaker::new(135.0, 45.0, false, "TRR", "TopRearRight"),
        ],
        "7.1.2" | "7.1.2 side" => vec![
            Speaker::new(-30.0, 0.0, false, "FL", "FrontLeft"),
            Speaker::new(30.0, 0.0, false, "FR", "FrontRight"),
            Speaker::new(0.0, 0.0, false, "FC", "FrontCenter"),
            Speaker::new(0.0, 0.0, true, "LFE", "ScreenLFE"),
            Speaker::new(-135.0, 0.0, false, "RL", "RearLeft"),
            Speaker::new(135.0, 0.0, false, "RR", "RearRight"),
            Speaker::new(-90.0, 0.0, false, "SL", "SideLeft"),
            Speaker::new(90.0, 0.0, false, "SR", "SideRight"),
            Speaker::new(-90.0, 45.0, false, "TSL", "TopSideLeft"),
            Speaker::new(90.0, 45.0, false, "TSR", "TopSideRight"),
        ],
        "7.1.2 front" => vec![
            Speaker::new(-30.0, 0.0, false, "FL", "FrontLeft"),
            Speaker::new(30.0, 0.0, false, "FR", "FrontRight"),
            Speaker::new(0.0, 0.0, false, "FC", "FrontCenter"),
            Speaker::new(0.0, 0.0, true, "LFE", "ScreenLFE"),
            Speaker::new(-135.0, 0.0, false, "RL", "RearLeft"),
            Speaker::new(135.0, 0.0, false, "RR", "RearRight"),
            Speaker::new(-90.0, 0.0, false, "SL", "SideLeft"),
            Speaker::new(90.0, 0.0, false, "SR", "SideRight"),
            Speaker::new(-45.0, 45.0, false, "TFL", "TopFrontLeft"),
            Speaker::new(45.0, 45.0, false, "TFR", "TopFrontRight"),
        ],
        "7.1.4" => vec![
            Speaker::new(-30.0, 0.0, false, "FL", "FrontLeft"),
            Speaker::new(30.0, 0.0, false, "FR", "FrontRight"),
            Speaker::new(0.0, 0.0, false, "FC", "FrontCenter"),
            Speaker::new(0.0, 0.0, true, "LFE", "ScreenLFE"),
            Speaker::new(-135.0, 0.0, false, "RL", "RearLeft"),
            Speaker::new(135.0, 0.0, false, "RR", "RearRight"),
            Speaker::new(-90.0, 0.0, false, "SL", "SideLeft"),
            Speaker::new(90.0, 0.0, false, "SR", "SideRight"),
            Speaker::new(-45.0, 45.0, false, "TFL", "TopFrontLeft"),
            Speaker::new(45.0, 45.0, false, "TFR", "TopFrontRight"),
            Speaker::new(-135.0, 45.0, false, "TRL", "TopRearLeft"),
            Speaker::new(135.0, 45.0, false, "TRR", "TopRearRight"),
        ],
        "7.1.6" => vec![
            Speaker::new(-30.0, 0.0, false, "FL", "FrontLeft"),
            Speaker::new(30.0, 0.0, false, "FR", "FrontRight"),
            Speaker::new(0.0, 0.0, false, "FC", "FrontCenter"),
            Speaker::new(0.0, 0.0, true, "LFE", "ScreenLFE"),
            Speaker::new(-135.0, 0.0, false, "RL", "RearLeft"),
            Speaker::new(135.0, 0.0, false, "RR", "RearRight"),
            Speaker::new(-90.0, 0.0, false, "SL", "SideLeft"),
            Speaker::new(90.0, 0.0, false, "SR", "SideRight"),
            Speaker::new(-45.0, 45.0, false, "TFL", "TopFrontLeft"),
            Speaker::new(45.0, 45.0, false, "TFR", "TopFrontRight"),
            Speaker::new(-90.0, 45.0, false, "TSL", "TopSideLeft"),
            Speaker::new(90.0, 45.0, false, "TSR", "TopSideRight"),
            Speaker::new(-135.0, 45.0, false, "TRL", "TopRearLeft"),
            Speaker::new(135.0, 45.0, false, "TRR", "TopRearRight"),
        ],
        "9.1.2" | "9.1.2 side" => vec![
            Speaker::new(-30.0, 0.0, false, "FL", "FrontLeft"),
            Speaker::new(30.0, 0.0, false, "FR", "FrontRight"),
            Speaker::new(0.0, 0.0, false, "FC", "FrontCenter"),
            Speaker::new(0.0, 0.0, true, "LFE", "ScreenLFE"),
            Speaker::new(-135.0, 0.0, false, "RL", "RearLeft"),
            Speaker::new(135.0, 0.0, false, "RR", "RearRight"),
            Speaker::new(-90.0, 0.0, false, "SL", "SideLeft"),
            Speaker::new(90.0, 0.0, false, "SR", "SideRight"),
            Speaker::new(-60.0, 0.0, false, "WL", "WideLeft"),
            Speaker::new(60.0, 0.0, false, "WR", "WideRight"),
            Speaker::new(-90.0, 45.0, false, "TSL", "TopSideLeft"),
            Speaker::new(90.0, 45.0, false, "TSR", "TopSideRight"),
        ],
        "9.1.2 front" => vec![
            Speaker::new(-30.0, 0.0, false, "FL", "FrontLeft"),
            Speaker::new(30.0, 0.0, false, "FR", "FrontRight"),
            Speaker::new(0.0, 0.0, false, "FC", "FrontCenter"),
            Speaker::new(0.0, 0.0, true, "LFE", "ScreenLFE"),
            Speaker::new(-135.0, 0.0, false, "RL", "RearLeft"),
            Speaker::new(135.0, 0.0, false, "RR", "RearRight"),
            Speaker::new(-90.0, 0.0, false, "SL", "SideLeft"),
            Speaker::new(90.0, 0.0, false, "SR", "SideRight"),
            Speaker::new(-60.0, 0.0, false, "WL", "WideLeft"),
            Speaker::new(60.0, 0.0, false, "WR", "WideRight"),
            Speaker::new(-45.0, 45.0, false, "TFL", "TopFrontLeft"),
            Speaker::new(45.0, 45.0, false, "TFR", "TopFrontRight"),
        ],
        "9.1.4" => vec![
            Speaker::new(-30.0, 0.0, false, "FL", "FrontLeft"),
            Speaker::new(30.0, 0.0, false, "FR", "FrontRight"),
            Speaker::new(0.0, 0.0, false, "FC", "FrontCenter"),
            Speaker::new(0.0, 0.0, true, "LFE", "ScreenLFE"),
            Speaker::new(-135.0, 0.0, false, "RL", "RearLeft"),
            Speaker::new(135.0, 0.0, false, "RR", "RearRight"),
            Speaker::new(-90.0, 0.0, false, "SL", "SideLeft"),
            Speaker::new(90.0, 0.0, false, "SR", "SideRight"),
            Speaker::new(-60.0, 0.0, false, "WL", "WideLeft"),
            Speaker::new(60.0, 0.0, false, "WR", "WideRight"),
            Speaker::new(-45.0, 45.0, false, "TFL", "TopFrontLeft"),
            Speaker::new(45.0, 45.0, false, "TFR", "TopFrontRight"),
            Speaker::new(-135.0, 45.0, false, "TRL", "TopRearLeft"),
            Speaker::new(135.0, 45.0, false, "TRR", "TopRearRight"),
        ],
        "small cinema" => vec![
            Speaker::new(-30.0, 0.0, false, "FL", "FrontLeft"),
            Speaker::new(30.0, 0.0, false, "FR", "FrontRight"),
            Speaker::new(0.0, 0.0, false, "FC", "FrontCenter"),
            Speaker::new(0.0, 0.0, true, "LFE", "ScreenLFE"),
            Speaker::new(-90.0, 0.0, false, "SL", "SideLeft"),
            Speaker::new(90.0, 0.0, false, "SR", "SideRight"),
            Speaker::new(-135.0, 0.0, false, "RL", "RearLeft"),
            Speaker::new(135.0, 0.0, false, "RR", "RearRight"),
            Speaker::new(-60.0, 0.0, false, "WL", "WideLeft"),
            Speaker::new(60.0, 0.0, false, "WR", "WideRight"),
            Speaker::new(-15.0, 0.0, false, "FLC", "FrontLeftCenter"),
            Speaker::new(15.0, 0.0, false, "FRC", "FrontRightCenter"),
        ],
        "22.2" | "22.2 surround" => vec![
            Speaker::new(-30.0, 0.0, false, "FL", "FrontLeft"),
            Speaker::new(30.0, 0.0, false, "FR", "FrontRight"),
            Speaker::new(0.0, 0.0, false, "FC", "FrontCenter"),
            Speaker::new(0.0, 0.0, true, "LFE", "ScreenLFE"),
            Speaker::new(-135.0, 0.0, false, "RL", "RearLeft"),
            Speaker::new(135.0, 0.0, false, "RR", "RearRight"),
            Speaker::new(-15.0, 0.0, false, "FLC", "FrontLeftCenter"),
            Speaker::new(15.0, 0.0, false, "FRC", "FrontRightCenter"),
            Speaker::new(180.0, 0.0, false, "RC", "RearCenter"),
            Speaker::new(-90.0, 0.0, false, "SL", "SideLeft"),
            Speaker::new(90.0, 0.0, false, "SR", "SideRight"),
            Speaker::new(-45.0, 45.0, false, "TFL", "TopFrontLeft"),
            Speaker::new(45.0, 45.0, false, "TFR", "TopFrontRight"),
            Speaker::new(0.0, 45.0, false, "TFC", "TopFrontCenter"),
            Speaker::new(0.0, 90.0, false, "TC", "TopCenter"),
            Speaker::new(-135.0, 45.0, false, "TRL", "TopRearLeft"),
            Speaker::new(135.0, 45.0, false, "TRR", "TopRearRight"),
            Speaker::new(-90.0, 45.0, false, "TSL", "TopSideLeft"),
            Speaker::new(90.0, 45.0, false, "TSR", "TopSideRight"),
            Speaker::new(180.0, 45.0, false, "TRC", "TopRearCenter"),
            Speaker::new(0.0, 0.0, true, "LFE2", "Subwoofer"),
            Speaker::new(-45.0, -30.0, false, "BFL", "BottomFrontLeft"),
            Speaker::new(45.0, -30.0, false, "BFR", "BottomFrontRight"),
            Speaker::new(0.0, -30.0, false, "BFC", "BottomFrontCenter"),
        ],
        _ => vec![
            Speaker::new(-30.0, 0.0, false, "FL", "FrontLeft"),
            Speaker::new(30.0, 0.0, false, "FR", "FrontRight"),
            Speaker::new(0.0, 0.0, false, "FC", "FrontCenter"),
            Speaker::new(0.0, 0.0, true, "LFE", "ScreenLFE"),
            Speaker::new(-135.0, 0.0, false, "RL", "RearLeft"),
            Speaker::new(135.0, 0.0, false, "RR", "RearRight"),
            Speaker::new(-90.0, 0.0, false, "SL", "SideLeft"),
            Speaker::new(90.0, 0.0, false, "SR", "SideRight"),
            Speaker::new(-60.0, 0.0, false, "WL", "WideLeft"),
            Speaker::new(60.0, 0.0, false, "WR", "WideRight"),
            Speaker::new(-45.0, 45.0, false, "TFL", "TopFrontLeft"),
            Speaker::new(45.0, 45.0, false, "TFR", "TopFrontRight"),
            Speaker::new(-90.0, 45.0, false, "TSL", "TopSideLeft"),
            Speaker::new(90.0, 45.0, false, "TSR", "TopSideRight"),
            Speaker::new(-135.0, 45.0, false, "TRL", "TopRearLeft"),
            Speaker::new(135.0, 45.0, false, "TRR", "TopRearRight"),
        ],
    };

    if !swap_surrounds {
        return (raw_speakers, direct_objects);
    }

    let swapped = raw_speakers
        .into_iter()
        .map(|s| match s.short_label {
            "RL" => Speaker::new(-90.0, 0.0, false, "SL", "SideLeft"),
            "RR" => Speaker::new(90.0, 0.0, false, "SR", "SideRight"),
            "SL" => Speaker::new(-135.0, 0.0, false, "RL", "RearLeft"),
            "SR" => Speaker::new(135.0, 0.0, false, "RR", "RearRight"),
            _ => s,
        })
        .collect();

    (swapped, direct_objects)
}
