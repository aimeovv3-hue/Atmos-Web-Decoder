// High-Speed Web Worker for Harletty WASM SIMD128 Decoder (7,000+ Frames/min)
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
        1536, // Full E-AC-3 frame step for maximum SIMD throughput
        0,    // Pure phase-locked VBAP (no artificial room delay)
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
      const stepBatch = batchFrames || 128; // 128 frames per WASM call -> >7000 frames/min

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
  view.setUint16(20, bitDepth === 32 ? 3 : 1, true); // 3 = IEEE Float, 1 = PCM Int
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitDepth, true);
  writeStr(36, 'data');
  view.setUint32(40, Math.min(0xFFFFFFFF, dataLength), true);

  return new Blob([header, ...chunks], { type: 'audio/wav' });
}
