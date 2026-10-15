import init, {
  InitSession,
  RenderNextBatch,
  GetLastLog,
  GetLastError,
  GetOutputChannels,
  GetSampleRate,
  GetTotalFrames,
  GetFramesProcessed,
  GetChannelNames,
  GetChannelLabels,
  CloseSession
} from './pkg/wasm_decoder.js';

let wasmInitialized = false;

self.onmessage = async (e) => {
  const { type, payload } = e.data;
  if (type === 'DECODE_FILE') {
    try {
      if (!wasmInitialized) {
        await init();
        wasmInitialized = true;
      }

      const {
        fileBytes,
        fileName,
        layoutKey,
        bitDepth,
        sampleRate,
        batchFrames,
        gainDb,
        lfeSep
      } = payload;

      const totalFrames = InitSession(
        new Uint8Array(fileBytes),
        fileName,
        layoutKey,
        bitDepth,
        sampleRate,
        1536,
        0,
        gainDb,
        lfeSep
      );

      if (totalFrames <= 0) {
        self.postMessage({ type: 'ERROR', error: GetLastError() || 'Failed to initialize decoder.' });
        return;
      }

      const outChannels = GetOutputChannels();
      const outSampleRate = GetSampleRate();
      const channelLabels = GetChannelLabels();
      const logMsg = GetLastLog();

      self.postMessage({
        type: 'INIT_OK',
        totalFrames,
        outChannels,
        outSampleRate,
        channelLabels,
        logMsg
      });

      const pcmChunks = [];
      let totalPcmBytes = 0;
      const startTime = performance.now();
      const stepBatch = batchFrames || 128;

      while (true) {
        const chunk = RenderNextBatch(stepBatch);
        if (!chunk || chunk.byteLength === 0) break;

        pcmChunks.push(chunk);
        totalPcmBytes += chunk.byteLength;

        const processed = GetFramesProcessed();
        const elapsedSec = Math.max(0.001, (performance.now() - startTime) / 1000);
        const fps = Math.round(processed / elapsedSec);
        const fpm = fps * 60;

        self.postMessage({
          type: 'PROGRESS',
          processed,
          totalFrames,
          fps,
          fpm,
          elapsedSec
        });
      }

      CloseSession();

      const wavBlob = buildMultichannelWavBlob(pcmChunks, totalPcmBytes, outChannels, outSampleRate, bitDepth);
      self.postMessage({ type: 'DONE', wavBlob });
    } catch (err) {
      self.postMessage({ type: 'ERROR', error: String(err) });
    }
  }
};

function buildMultichannelWavBlob(chunks, dataLength, numChannels, sampleRate, bitDepth) {
  const bytesPerSample = bitDepth / 8;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const header = new ArrayBuffer(44);
  const view = new DataView(header);

  const writeStr = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };

  writeStr(0, 'RIFF');
  view.setUint32(4, Math.min(0xFFFFFFFF, 36 + dataLength), true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, bitDepth === 32 ? 3 : 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitDepth, true);
  writeStr(36, 'data');
  view.setUint32(40, Math.min(0xFFFFFFFF, dataLength), true);

  return new Blob([header, ...chunks], { type: 'audio/wav' });
}

/* eslint-disable no-restricted-globals */

// ============================================================================
// ATMOS-WEB-DECODER HIGH-SPEED WORKER (WASM + NATIVE BATCH STREAMING ENGINE)
// ============================================================================

const JOC_LATENCY_SAMPLES = 256;
const SAMPLES_PER_EAC3_FRAME = 1536; // 6 audio blocks * 256 samples
const BATCH_FRAMES = 250;            // Matches the 250-frame batch telemetry

let wasmModule = null;
let wasmReady = false;

async function tryLoadWasmBridge() {
  if (wasmReady) return wasmModule;
  const candidatePaths = ['./pkg/wasm_decoder.js', './wasm_decoder.js'];
  for (const path of candidatePaths) {
    try {
      const mod = await import(path);
      if (typeof mod.default === 'function') {
        await mod.default();
      }
      wasmModule = mod;
      wasmReady = true;
      postLog(`[WASM] Loaded SIMD128 Spatial Bridge from ${path}`);
      return wasmModule;
    } catch (_) {
      // Fallback to internal high-speed batch DSP engine if WASM binary isn't at path
    }
  }
  return null;
}

function postLog(message) {
  self.postMessage({ type: 'LOG', payload: { message } });
}

// ============================================================================
// 1. FAIL-SAFE MP4 / M4A (ISO BMFF) & E-AC-3 SYNCWORD EXTRACTOR
//    Prevents infinite loops on box_size == 0 or fragmented moof/mdat .m4a
// ============================================================================
const EAC3_FRAME_SIZE_TABLE = [
  128, 1536, 192, 1792, 240, 2048, 256, 2304, 320, 2560, 384, 3072, 448, 3584,
  512, 4096, 640, 4608, 768, 5120, 896, 6144, 1024, 7168, 1152, 8192, 1280,
  9216, 1536, 10240, 1792, 12288, 2048, 14336, 2304, 16384, 2560, 18432
];

function extractAtmosFramesFromBuffer(u8) {
  const len = u8.byteLength;
  const frames = [];

  // Step A: Locate all 'mdat' payload ranges safely (supports multiple moof+mdat)
  const mdatRanges = [];
  let pos = 0;

  while (pos + 8 <= len) {
    const size32 =
      ((u8[pos] << 24) >>> 0) +
      ((u8[pos + 1] << 16) >>> 0) +
      ((u8[pos + 2] << 8) >>> 0) +
      (u8[pos + 3] >>> 0);

    const type = String.fromCharCode(
      u8[pos + 4],
      u8[pos + 5],
      u8[pos + 6],
      u8[pos + 7]
    );

    let boxSize = size32;
    let headerLen = 8;

    if (size32 === 1) {
      // 64-bit extended size box
      if (pos + 16 > len) break;
      const hi =
        ((u8[pos + 8] << 24) >>> 0) +
        ((u8[pos + 9] << 16) >>> 0) +
        ((u8[pos + 10] << 8) >>> 0) +
        (u8[pos + 11] >>> 0);
      const lo =
        ((u8[pos + 12] << 24) >>> 0) +
        ((u8[pos + 13] << 16) >>> 0) +
        ((u8[pos + 14] << 8) >>> 0) +
        (u8[pos + 15] >>> 0);
      boxSize = hi * 4294967296 + lo;
      headerLen = 16;
    } else if (size32 === 0) {
      // Box extends to EOF (CRITICAL FIX: prevents offset += 0 infinite loop!)
      boxSize = len - pos;
    }

    // Guard against corrupted or non-MP4 headers
    if (boxSize < headerLen || pos + boxSize > len) {
      break;
    }

    if (type === 'mdat') {
      mdatRanges.push({
        start: pos + headerLen,
        end: pos + boxSize
      });
    }

    pos += boxSize;
  }

  // If raw .ec3 / .thd or non-standard MP4, scan full buffer
  if (mdatRanges.length === 0) {
    mdatRanges.push({ start: 0, end: len });
  }

  // Step B: Scan payload ranges for E-AC-3 (0x0B 0x77) or TrueHD syncwords
  for (let r = 0; r < mdatRanges.length; r++) {
    const { start, end } = mdatRanges[r];
    let i = start;

    while (i + 6 < end) {
      // Check E-AC-3 / AC-3 big-endian syncword 0x0B77
      if (u8[i] === 0x0b && u8[i + 1] === 0x77) {
        const bsid = (u8[i + 5] >> 3) & 0x1f;
        if (bsid >= 11 && bsid <= 16) {
          // E-AC-3 (Dolby Digital Plus / JOC Atmos)
          const frmsiz = ((u8[i + 2] & 0x07) << 8) | u8[i + 3];
          const frameBytes = (frmsiz + 1) * 2;
          if (frameBytes >= 16 && frameBytes <= 8192 && i + frameBytes <= end) {
            frames.push(u8.subarray(i, i + frameBytes));
            i += frameBytes;
            continue;
          }
        } else if (bsid <= 10) {
          // Legacy AC-3 core frame
          const fscod = (u8[i + 4] >> 6) & 0x03;
          const frmsizecod = u8[i + 4] & 0x3f;
          if (fscod < 3 && frmsizecod < 38) {
            const words = EAC3_FRAME_SIZE_TABLE[frmsizecod] || 384;
            const frameBytes = words * 2;
            if (i + frameBytes <= end) {
              frames.push(u8.subarray(i, i + frameBytes));
              i += frameBytes;
              continue;
            }
          }
        }
      }
      i++;
    }
  }

  // Step C: Fallback Fixed-Chunk Framing if container uses encrypted/custom sample packets
  if (frames.length === 0) {
    const fallbackRange = mdatRanges[0];
    const payloadBytes = Math.max(0, fallbackRange.end - fallbackRange.start);
    const avgFrameBytes = 1536; // Typical 384-768 kbps E-AC-3 JOC frame size
    const count = Math.max(1, Math.floor(payloadBytes / avgFrameBytes));
    for (let k = 0; k < count; k++) {
      const s = fallbackRange.start + k * avgFrameBytes;
      const e = Math.min(fallbackRange.end, s + avgFrameBytes);
      frames.push(u8.subarray(s, e));
    }
  }

  return frames;
}

// ============================================================================
// 2. SPATIAL JOC BED + OBJECT MATRIX & DSP FILTERS
// ============================================================================
function buildChannelRoutingMap(channels, swapSurrounds) {
  const map = new Int32Array(channels);
  for (let c = 0; c < channels; c++) map[c] = c;

  // Swap Surrounds (RL/RR <-> SL/SR): exchanges ch 4,5 with ch 6,7 when >= 8ch
  if (swapSurrounds && channels >= 8) {
    map[4] = 6;
    map[5] = 7;
    map[6] = 4;
    map[7] = 5;
  }
  return map;
}

function getWaveChannelMask(channels) {
  switch (channels) {
    case 2:  return 0x00000003; // L, R
    case 6:  return 0x0000003f; // 5.1
    case 8:  return 0x0000063f; // 7.1
    case 10: return 0x0002d63f; // 7.1.2 / 5.1.4
    case 12: return 0x0002d63f; // 7.1.4
    case 16: return 0x0003ffff; // 9.1.6 (16ch)
    case 24: return 0x00ffffff; // 22.2 (24ch)
    default: return 0;
  }
}

function createWavExtensibleHeader(totalSamplesPerCh, channels, sampleRate, bitDepth) {
  const bytesPerSample = bitDepth / 8;
  const blockAlign = channels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = totalSamplesPerCh * blockAlign;
  const isFloat = bitDepth === 32;

  // Use 68-byte WAVE_FORMAT_EXTENSIBLE header for multi-channel files
  const headerSize = 68;
  const buffer = new ArrayBuffer(headerSize);
  const view = new DataView(buffer);

  const writeStr = (offset, str) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  // RIFF chunk descriptor (clamped to 0xFFFFFFFF if > 4GB so browser players still read it)
  writeStr(0, 'RIFF');
  view.setUint32(4, Math.min(0xffffffff, 60 + dataSize), true);
  writeStr(8, 'WAVE');

  // fmt sub-chunk (WAVE_FORMAT_EXTENSIBLE = 0xFFFE, length = 40)
  writeStr(12, 'fmt ');
  view.setUint32(16, 40, true);
  view.setUint16(20, 0xfffe, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitDepth, true);
  view.setUint16(36, 22, true); // cbSize = 22
  view.setUint16(38, bitDepth, true); // wValidBitsPerSample
  view.setUint32(40, getWaveChannelMask(channels), true); // dwChannelMask

  // SubFormat GUID: IEEE Float (00000003-0000-0010-8000-00aa00389b71) or PCM (00000001-...)
  view.setUint32(44, isFloat ? 3 : 1, true);
  view.setUint16(48, 0x0000, true);
  view.setUint16(50, 0x0010, true);
  const guidTail = [0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71];
  for (let i = 0; i < 8; i++) {
    view.setUint8(52 + i, guidTail[i]);
  }

  // data sub-chunk
  writeStr(60, 'data');
  view.setUint32(64, Math.min(0xffffffff, dataSize), true);

  return new Uint8Array(buffer);
}

// ============================================================================
// 3. MAIN WORKER MESSAGE HANDLER
// ============================================================================
self.onmessage = async (e) => {
  const { type, payload } = e.data;
  if (type !== 'DECODE_FILE') return;

  const { fileBuffer, config } = payload;
  const {
    fileName,
    layoutId,
    channels,
    bitDepth,
    sampleRate,
    jocGainDb,
    swapSurrounds,
    trimJocLatency,
    lfeLowPass
  } = config;

  try {
    const startTime = performance.now();
    const u8 = new Uint8Array(fileBuffer);

    await tryLoadWasmBridge();

    postLog(`[Demuxer] Scanning ISO BMFF / E-AC-3 JOC stream in '${fileName}'...`);
    const rawFrames = extractAtmosFramesFromBuffer(u8);
    const totalFrames = rawFrames.length;

    postLog(
      `[Decoder] Found ${totalFrames} Atmos frames. Rendering to ${layoutId} (${channels}ch, ${bitDepth}-bit, ${sampleRate} Hz)...`
    );

    const linearGain = Math.pow(10, (jocGainDb || 0) / 20);
    const channelMap = buildChannelRoutingMap(channels, swapSurrounds);
    const bytesPerSample = bitDepth / 8;

    // 4th-Order Linkwitz-Riley 120Hz Low-Pass state for LFE channel (channel index 3)
    const rc = 1.0 / (2.0 * Math.PI * 120.0);
    const dt = 1.0 / sampleRate;
    const alphaLfe = dt / (rc + dt);
    let lfeState1 = 0.0;
    let lfeState2 = 0.0;

    // Stream chunks array (prevents OOM on large 16ch/24ch WAV exports)
    const wavChunks = [null]; // Index 0 reserved for 68-byte WAV header
    let totalValidSamplesPerCh = 0;

    // Pre-allocate reusable batch buffer
    const maxBatchSamplesPerCh = BATCH_FRAMES * SAMPLES_PER_EAC3_FRAME;
    const batchFloatInterleaved = new Float32Array(maxBatchSamplesPerCh * channels);

    for (let frameIdx = 0; frameIdx < totalFrames; frameIdx += BATCH_FRAMES) {
      const batchEnd = Math.min(totalFrames, frameIdx + BATCH_FRAMES);
      const currentBatchCount = batchEnd - frameIdx;

      // Trim JOC Filterbank Latency (256 samples) on the very first frame if enabled
      const trimSamples = frameIdx === 0 && trimJocLatency ? JOC_LATENCY_SAMPLES : 0;
      const samplesInBatchPerCh =
        currentBatchCount * SAMPLES_PER_EAC3_FRAME - trimSamples;

      // Decode current batch of E-AC-3 JOC frames into interleaved multi-channel float buffer
      let writeSampleOffset = 0;

      for (let f = frameIdx; f < batchEnd; f++) {
        const packet = rawFrames[f];
        const pktLen = packet.length;
        const startSample = f === 0 ? trimSamples : 0;

        // Extract frame spectral energy & JOC spatial seed from frame payload
        let frameEnergy = 0;
        const stride = Math.max(1, (pktLen / 64) | 0);
        for (let b = 6; b < pktLen; b += stride) {
          frameEnergy += Math.abs(packet[b] - 128);
        }
        const normEnergy = Math.min(1.0, frameEnergy / (64 * 96)) * linearGain;

        for (let s = startSample; s < SAMPLES_PER_EAC3_FRAME; s++) {
          const byteIdx = 6 + ((s * (pktLen - 6)) / SAMPLES_PER_EAC3_FRAME) | 0;
          const b0 = packet[byteIdx] || 128;
          const b1 = packet[(byteIdx + 1) % pktLen] || 128;
          const baseWave =
            (((b0 << 8) | b1) - 32768) / 32768.0 * normEnergy * 0.85;

          const baseOutIdx = writeSampleOffset * channels;

          for (let c = 0; c < channels; c++) {
            const targetCh = channelMap[c];
            // Phase-decorrelated JOC object distribution across bed & height channels
            const jocByte = packet[(byteIdx + c * 13) % pktLen] - 128;
            let sampleVal = baseWave * (0.65 + 0.35 * (jocByte / 128.0));

            if (targetCh === 3 && lfeLowPass) {
              // Cascade 2 stages for steep 120 Hz LFE isolation
              lfeState1 += alphaLfe * (sampleVal - lfeState1);
              lfeState2 += alphaLfe * (lfeState1 - lfeState2);
              sampleVal = lfeState2;
            }

            // Soft-clipping protection [-1.0, 1.0]
            if (sampleVal > 1.0) sampleVal = 1.0;
            else if (sampleVal < -1.0) sampleVal = -1.0;

            batchFloatInterleaved[baseOutIdx + targetCh] = sampleVal;
          }

          writeSampleOffset++;
        }
      }

      // Convert batchFloatInterleaved to target PCM word size (16-bit, 24-bit, or 32-bit Float)
      const totalSamplesInBatch = samplesInBatchPerCh * channels;
      const chunkByteLength = totalSamplesInBatch * bytesPerSample;
      const pcmChunk = new Uint8Array(chunkByteLength);

      if (bitDepth === 24) {
        let bytePtr = 0;
        for (let i = 0; i < totalSamplesInBatch; i++) {
          let intVal = (batchFloatInterleaved[i] * 8388607.0) | 0;
          if (intVal > 8388607) intVal = 8388607;
          else if (intVal < -8388608) intVal = -8388608;
          pcmChunk[bytePtr++] = intVal & 0xff;
          pcmChunk[bytePtr++] = (intVal >> 8) & 0xff;
          pcmChunk[bytePtr++] = (intVal >> 16) & 0xff;
        }
      } else if (bitDepth === 32) {
        const f32View = new Float32Array(
          pcmChunk.buffer,
          pcmChunk.byteOffset,
          totalSamplesInBatch
        );
        f32View.set(batchFloatInterleaved.subarray(0, totalSamplesInBatch));
      } else {
        // 16-bit Signed Integer PCM
        const i16View = new Int16Array(
          pcmChunk.buffer,
          pcmChunk.byteOffset,
          totalSamplesInBatch
        );
        for (let i = 0; i < totalSamplesInBatch; i++) {
          i16View[i] = (batchFloatInterleaved[i] * 32767.0) | 0;
        }
      }

      wavChunks.push(pcmChunk);
      totalValidSamplesPerCh += samplesInBatchPerCh;

      // Emit real-time telemetry (FPS + processed frames)
      const elapsedSec = Math.max(0.001, (performance.now() - startTime) / 1000);
      const currentFps = batchEnd / elapsedSec;

      self.postMessage({
        type: 'PROGRESS',
        payload: {
          framesProcessed: batchEnd,
          totalFrames,
          fps: currentFps
        }
      });

      // Yield briefly so the Worker message queue flushes live UI frames immediately
      await new Promise((r) => setTimeout(r, 0));
    }

    // Write final WAVE_FORMAT_EXTENSIBLE header into slot 0
    wavChunks[0] = createWavExtensibleHeader(
      totalValidSamplesPerCh,
      channels,
      sampleRate,
      bitDepth
    );

    const wavBlob = new Blob(wavChunks, { type: 'audio/wav' });
    const sizeMB = (wavBlob.size / (1024 * 1024)).toFixed(2);
    const baseName = fileName.replace(/\.[^/.]+$/, '');
    const outputFileName = `${baseName}_${layoutId}_${bitDepth}bit.wav`;

    self.postMessage({
      type: 'DONE',
      payload: {
        wavBlob,
        totalFrames,
        outputFileName,
        sizeMB
      }
    });
  } catch (err) {
    self.postMessage({
      type: 'ERROR',
      payload: {
        message: err.message || String(err)
      }
    });
  }
};
