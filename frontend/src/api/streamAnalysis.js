import { isAnalysisResponse } from "./analyze.js";
import {
  MAX_CAPTURE_SECONDS,
  MicrophoneCaptureError,
  microphoneError,
  requestMicrophone,
} from "../audio/microphone.js";

const CHUNK_SECONDS = 0.25;
const MAX_SOCKET_BUFFER_BYTES = 512 * 1024;
const SOCKET_OPEN_TIMEOUT_MS = 5000;
const LEVEL_UPDATE_INTERVAL_MS = 100;

const STREAM_ERROR_COPY = {
  AUDIO_TOO_LONG: "The live sample reached the service duration limit.",
  INVALID_AUDIO: "The service could not interpret the live microphone samples.",
  MISSING_AUDIO: "No microphone audio reached the service. Please retry.",
  MODEL_UNAVAILABLE: "The analyzer is still starting. Try again in a moment.",
  PAYLOAD_TOO_LARGE: "The live audio stream exceeded the service size limit.",
  SERVICE_BUSY: "The analyzer is busy. Please start a new live session shortly.",
  WS_IDLE_TIMEOUT: "The live stream was idle for too long.",
  WS_PROTOCOL_ERROR: "The service rejected the live-stream protocol.",
  WS_SESSION_TIMEOUT: "The live session reached its time limit.",
};

export class StreamAnalysisError extends Error {
  constructor(message, { code = "STREAM_ERROR", requestId = null } = {}) {
    super(message);
    this.name = "StreamAnalysisError";
    this.code = code;
    this.requestId = requestId;
  }
}

export function analysisWebSocketUrl(location = window.location) {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${location.host}/api/ws/analyze`;
}

function isPredictionMessage(payload) {
  return Boolean(
    payload?.type === "prediction" &&
      Number.isInteger(payload.sequence) &&
      payload.sequence > 0 &&
      typeof payload.is_final === "boolean" &&
      isAnalysisResponse(payload),
  );
}

function parseServiceError(payload) {
  const error = payload?.error;
  const validCode =
    typeof error?.code === "string" && error.code.length > 0 && error.code.length <= 64;
  const validMessage =
    error?.message === undefined ||
    (typeof error.message === "string" && error.message.length <= 500);
  const validRequestId =
    error?.request_id === undefined ||
    error.request_id === null ||
    (typeof error.request_id === "string" && error.request_id.length <= 128);
  if (!validCode || !validMessage || !validRequestId) return null;
  return {
    code: error.code,
    message: error.message,
    requestId: error.request_id || null,
  };
}

function rms(samples) {
  if (!samples.length) return 0;
  let sum = 0;
  for (const sample of samples) sum += sample * sample;
  return Math.min(1, Math.sqrt(sum / samples.length) * 3.5);
}

function wipeChunks(chunks) {
  for (const chunk of chunks) chunk.fill(0);
  chunks.length = 0;
}

class PcmMicrophoneCapture {
  constructor({ isCancelled, onSamples, onLevel }) {
    this.isCancelled = isCancelled;
    this.onSamples = onSamples;
    this.onLevel = onLevel;
    this.stream = null;
    this.context = null;
    this.source = null;
    this.worklet = null;
    this.silence = null;
    this.flushResolver = null;
    this.latestLevel = 0;
    this.levelTimer = null;
  }

  async prepare() {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass || typeof AudioWorkletNode === "undefined") {
      throw new MicrophoneCaptureError(
        "Live microphone streaming is not supported by this browser.",
        "LIVE_UNSUPPORTED",
      );
    }

    // Create and resume under the original click gesture. In particular, this
    // avoids delayed/resampled AudioWorklet capture on some WebKit versions.
    this.context = new AudioContextClass({ latencyHint: "interactive" });
    const resumeResult = Promise.resolve(this.context.resume()).then(
      () => null,
      (error) => error,
    );
    try {
      this.stream = await requestMicrophone();
      if (this.isCancelled()) {
        throw new MicrophoneCaptureError("Live capture cancelled.", "LIVE_CANCELLED");
      }
      const resumeError = await resumeResult;
      if (resumeError) throw resumeError;
      if (this.context.sampleRate < 8000 || this.context.sampleRate > 96000) {
        throw new MicrophoneCaptureError(
          "The browser selected an unsupported microphone sample rate.",
          "UNSUPPORTED_SAMPLE_RATE",
        );
      }
      await this.context.audioWorklet.addModule("/pcm-capture-worklet.js");
      if (this.isCancelled()) {
        throw new MicrophoneCaptureError("Live capture cancelled.", "LIVE_CANCELLED");
      }
      this.source = this.context.createMediaStreamSource(this.stream);
      this.worklet = new AudioWorkletNode(this.context, "pcm-capture", {
        channelCount: 1,
        channelCountMode: "explicit",
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
      });
      this.silence = this.context.createGain();
      this.silence.gain.value = 0;
      this.worklet.connect(this.silence);
      this.silence.connect(this.context.destination);
      this.worklet.port.onmessage = (event) => {
        if (event.data?.type === "samples" && event.data.buffer) {
          const samples = new Float32Array(event.data.buffer);
          this.scheduleLevel(rms(samples));
          this.onSamples(samples);
        } else if (event.data?.type === "flush-complete") {
          this.flushResolver?.();
          this.flushResolver = null;
        }
      };
      return this.context.sampleRate;
    } catch (error) {
      await this.stop();
      throw microphoneError(error);
    }
  }

  async start() {
    if (this.isCancelled()) {
      throw new MicrophoneCaptureError("Live capture cancelled.", "LIVE_CANCELLED");
    }
    this.source.connect(this.worklet);
    if (this.context.state !== "running") await this.context.resume();
    if (this.context.state !== "running") {
      throw new MicrophoneCaptureError(
        "The browser suspended live microphone processing. Please retry.",
        "AUDIO_CONTEXT_SUSPENDED",
      );
    }
  }

  scheduleLevel(level) {
    this.latestLevel = level;
    if (this.levelTimer !== null) return;
    this.levelTimer = window.setTimeout(() => {
      this.levelTimer = null;
      this.onLevel(this.latestLevel);
    }, LEVEL_UPDATE_INTERVAL_MS);
  }

  flush() {
    if (!this.worklet) return Promise.resolve();
    return new Promise((resolve) => {
      const timeout = window.setTimeout(() => {
        this.flushResolver = null;
        resolve();
      }, 150);
      this.flushResolver = () => {
        window.clearTimeout(timeout);
        resolve();
      };
      this.worklet.port.postMessage({ type: "flush" });
    });
  }

  async stop() {
    window.clearTimeout(this.levelTimer);
    this.levelTimer = null;
    this.latestLevel = 0;
    this.worklet?.port.postMessage({ type: "dispose" });
    if (this.worklet) this.worklet.port.onmessage = null;
    for (const node of [this.source, this.worklet, this.silence]) {
      try {
        node?.disconnect();
      } catch {
        // A partially initialized node may already be disconnected.
      }
    }
    for (const track of this.stream?.getTracks?.() || []) track.stop();
    if (this.context && this.context.state !== "closed") {
      try {
        await this.context.close();
      } catch {
        // Browser teardown is best effort.
      }
    }
    this.stream = null;
    this.context = null;
    this.source = null;
    this.worklet = null;
    this.silence = null;
  }
}

export class LiveAnalysisSession {
  constructor({ onError, onPrediction, onState, onStats, webSocketFactory } = {}) {
    this.onError = onError || (() => {});
    this.onPrediction = onPrediction || (() => {});
    this.onState = onState || (() => {});
    this.onStats = onStats || (() => {});
    this.webSocketFactory = webSocketFactory || ((url) => new WebSocket(url));
    this.capture = null;
    this.socket = null;
    this.sampleRate = 0;
    this.pending = [];
    this.pendingSamples = 0;
    this.samplesSent = 0;
    this.chunksSent = 0;
    this.bytesSent = 0;
    this.lastSequence = 0;
    this.phase = "idle";
    this.cancelled = false;
    this.failed = false;
    this.finalReceived = false;
    this.autoFinishing = false;
    this.level = 0;
    this.abortOpen = null;
  }

  async start(contactId = "") {
    if (this.phase !== "idle") return;
    this.phase = "requesting";
    this.onState("requesting");
    this.capture = new PcmMicrophoneCapture({
      isCancelled: () => this.cancelled,
      onLevel: (level) => this.emitStats(level),
      onSamples: (samples) => this.acceptSamples(samples),
    });

    try {
      this.sampleRate = await this.capture.prepare();
      this.ensureActive();
      this.phase = "connecting";
      this.onState("connecting");
      this.socket = this.webSocketFactory(analysisWebSocketUrl());
      this.socket.binaryType = "arraybuffer";
      await this.waitForOpen();
      this.ensureActive();
      this.attachSocketHandlers();
      const start = {
        type: "start",
        encoding: "pcm_f32le",
        sample_rate: this.sampleRate,
        channels: 1,
      };
      if (contactId.trim()) start.contact_id = contactId.trim();
      try {
        this.socket.send(JSON.stringify(start));
      } catch {
        throw new StreamAnalysisError("The live connection was interrupted.", {
          code: "STREAM_CONNECTION_ERROR",
        });
      }
      await this.capture.start();
      this.ensureActive();
      this.phase = "streaming";
      this.onState("streaming");
      this.emitStats(0);
    } catch (error) {
      await this.fail(
        error instanceof StreamAnalysisError || error instanceof MicrophoneCaptureError
          ? error
          : microphoneError(error),
      );
      throw error;
    }
  }

  ensureActive() {
    if (this.cancelled) {
      throw new MicrophoneCaptureError("Live capture cancelled.", "LIVE_CANCELLED");
    }
  }

  waitForOpen() {
    return new Promise((resolve, reject) => {
      const cleanup = () => {
        window.clearTimeout(timer);
        this.socket.removeEventListener("open", opened);
        this.socket.removeEventListener("error", errored);
        this.socket.removeEventListener("close", closed);
        this.abortOpen = null;
      };
      const rejectWith = (error) => {
        cleanup();
        reject(error);
      };
      const timer = window.setTimeout(() => {
        rejectWith(
          new StreamAnalysisError("The live connection timed out.", {
            code: "STREAM_CONNECT_TIMEOUT",
          }),
        );
      }, SOCKET_OPEN_TIMEOUT_MS);
      const opened = () => {
        cleanup();
        resolve();
      };
      const errored = () => {
        rejectWith(
          new StreamAnalysisError("The live connection could not be established.", {
            code: "STREAM_CONNECT_FAILED",
          }),
        );
      };
      const closed = () => {
        rejectWith(
          new StreamAnalysisError("The live connection closed while starting.", {
            code: "STREAM_CONNECT_FAILED",
          }),
        );
      };
      this.abortOpen = () => {
        rejectWith(new MicrophoneCaptureError("Live capture cancelled.", "LIVE_CANCELLED"));
      };
      this.socket.addEventListener("open", opened, { once: true });
      this.socket.addEventListener("error", errored, { once: true });
      this.socket.addEventListener("close", closed, { once: true });
    });
  }

  attachSocketHandlers() {
    this.socket.addEventListener("message", (event) => this.handleMessage(event));
    this.socket.addEventListener("error", () => {
      if (!this.cancelled && !this.finalReceived) {
        void this.fail(
          new StreamAnalysisError("The live connection was interrupted.", {
            code: "STREAM_CONNECTION_ERROR",
          }),
        );
      }
    });
    this.socket.addEventListener("close", (event) => {
      if (!this.cancelled && !this.finalReceived && !this.failed) {
        const code = event.code === 1006 ? "STREAM_CONNECTION_LOST" : `WS_CLOSE_${event.code}`;
        void this.fail(
          new StreamAnalysisError("The live connection closed before a final result.", {
            code,
          }),
        );
      }
    });
  }

  handleMessage(event) {
    if (this.cancelled || this.failed || this.finalReceived) return;
    let payload;
    try {
      payload = JSON.parse(event.data);
    } catch {
      void this.fail(
        new StreamAnalysisError("The service sent an invalid live response.", {
          code: "INVALID_STREAM_RESPONSE",
        }),
      );
      return;
    }
    if (payload?.type === "pong") return;
    if (payload?.type === "error") {
      const serviceError = parseServiceError(payload);
      if (!serviceError) {
        void this.fail(
          new StreamAnalysisError("The service sent an invalid live response.", {
            code: "INVALID_STREAM_RESPONSE",
          }),
        );
        return;
      }
      void this.fail(
        new StreamAnalysisError(
          STREAM_ERROR_COPY[serviceError.code] ||
            serviceError.message ||
            "Live analysis failed.",
          { code: serviceError.code, requestId: serviceError.requestId },
        ),
      );
      return;
    }
    if (!isPredictionMessage(payload)) {
      void this.fail(
        new StreamAnalysisError("The service sent an unexpected live response.", {
          code: "INVALID_STREAM_RESPONSE",
        }),
      );
      return;
    }
    if (payload.sequence <= this.lastSequence) return;
    if (payload.is_final) {
      this.finalReceived = true;
      this.phase = "complete";
    }
    this.lastSequence = payload.sequence;
    this.onPrediction(payload);
    if (payload.is_final) {
      wipeChunks(this.pending);
      this.pendingSamples = 0;
      this.onState("complete");
      void this.capture?.stop();
    }
  }

  acceptSamples(samples) {
    if (this.phase !== "streaming" && this.phase !== "finalizing") {
      samples.fill(0);
      return;
    }
    this.pending.push(samples);
    this.pendingSamples += samples.length;
    if (this.pendingSamples >= Math.round(this.sampleRate * CHUNK_SECONDS)) {
      this.sendPending();
    }
  }

  sendPending() {
    if (!this.pendingSamples) return true;
    if (this.socket?.readyState !== WebSocket.OPEN) return false;
    const payload = new ArrayBuffer(this.pendingSamples * 4);
    const view = new DataView(payload);
    let offset = 0;
    for (const chunk of this.pending) {
      for (const sample of chunk) {
        view.setFloat32(offset, Math.max(-1, Math.min(1, sample)), true);
        offset += 4;
      }
    }
    wipeChunks(this.pending);
    const sampleCount = this.pendingSamples;
    this.pendingSamples = 0;

    if (this.socket.bufferedAmount + payload.byteLength > MAX_SOCKET_BUFFER_BYTES) {
      new Uint8Array(payload).fill(0);
      void this.fail(
        new StreamAnalysisError(
          "The network is too slow for privacy-safe live streaming. Please retry on a stable connection.",
          { code: "STREAM_BACKPRESSURE" },
        ),
      );
      return false;
    }
    try {
      this.socket.send(payload);
    } catch {
      new Uint8Array(payload).fill(0);
      void this.fail(
        new StreamAnalysisError("The live connection was interrupted.", {
          code: "STREAM_CONNECTION_ERROR",
        }),
      );
      return false;
    }
    this.samplesSent += sampleCount;
    this.chunksSent += 1;
    this.bytesSent += payload.byteLength;
    this.emitStats();
    if (
      this.samplesSent / this.sampleRate >= MAX_CAPTURE_SECONDS &&
      !this.autoFinishing
    ) {
      this.autoFinishing = true;
      void this.finish();
    }
    return true;
  }

  emitStats(level) {
    if (Number.isFinite(level)) this.level = level;
    this.onStats({
      bytes: this.bytesSent,
      chunks: this.chunksSent,
      elapsed: this.sampleRate ? this.samplesSent / this.sampleRate : 0,
      level: this.level,
      sampleRate: this.sampleRate,
    });
  }

  async finish() {
    if (this.phase !== "streaming") return;
    this.phase = "finalizing";
    this.onState("finalizing");
    await this.capture?.flush();
    if (this.cancelled || this.failed || this.finalReceived) return;
    await this.capture?.stop();
    if (this.cancelled || this.failed || this.finalReceived) return;
    const sent = this.sendPending();
    if (!sent || this.failed) return;
    if (this.socket?.readyState === WebSocket.OPEN) {
      try {
        this.socket.send(JSON.stringify({ type: "end" }));
      } catch {
        await this.fail(
          new StreamAnalysisError("The connection closed before audio could be finalized.", {
            code: "STREAM_CONNECTION_LOST",
          }),
        );
      }
    } else {
      await this.fail(
        new StreamAnalysisError("The connection closed before audio could be finalized.", {
          code: "STREAM_CONNECTION_LOST",
        }),
      );
    }
  }

  async fail(error) {
    if (this.failed || this.cancelled || this.finalReceived) return;
    this.failed = true;
    this.phase = "error";
    wipeChunks(this.pending);
    this.pendingSamples = 0;
    await this.capture?.stop();
    if (this.socket && this.socket.readyState < WebSocket.CLOSING) this.socket.close(1000);
    if (this.cancelled || this.finalReceived) return;
    this.onError(error);
    this.onState("error");
  }

  async cancel() {
    if (this.cancelled) return;
    this.cancelled = true;
    this.phase = "cancelled";
    this.abortOpen?.();
    wipeChunks(this.pending);
    this.pendingSamples = 0;
    await this.capture?.stop();
    if (this.socket && this.socket.readyState < WebSocket.CLOSING) this.socket.close(1000);
    this.onState("cancelled");
  }
}
