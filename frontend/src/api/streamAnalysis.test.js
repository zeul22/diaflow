import { beforeEach, describe, expect, it, vi } from "vitest";

import { LiveAnalysisSession, analysisWebSocketUrl } from "./streamAnalysis.js";

const CONTACT_ID = "123e4567-e89b-12d3-a456-426614174000";

function prediction(sequence, isFinal) {
  return {
    type: "prediction",
    sequence,
    is_final: isFinal,
    contact_id: CONTACT_ID,
    gender: { prediction: "unknown", confidence: 0 },
    age_bracket: { prediction: "31-45", confidence: 0.41 },
    processing_ms: 91,
    audio_quality: "degraded",
  };
}

function persistenceReceipt(overrides = {}) {
  return {
    mode: "result_and_audio",
    status: "pending",
    chunks_received: 0,
    chunks_stored: 0,
    segments_stored: 0,
    bytes_stored: 0,
    audio_expires_at: "2026-08-25T10:00:00Z",
    result_expires_at: "2026-09-24T10:00:00Z",
    ...overrides,
  };
}

class FakeAudioNode {
  constructor() {
    this.connect = vi.fn(() => this);
    this.disconnect = vi.fn();
  }
}

class FakeAudioWorkletNode extends FakeAudioNode {
  static instances = [];

  constructor(context, name, options) {
    super();
    this.context = context;
    this.name = name;
    this.options = options;
    this.port = {
      onmessage: null,
      postMessage: vi.fn((message) => {
        if (message?.type === "flush") {
          queueMicrotask(() => {
            this.port.onmessage?.({ data: { type: "flush-complete" } });
          });
        }
      }),
    };
    FakeAudioWorkletNode.instances.push(this);
  }

  emitSamples(samples) {
    this.port.onmessage?.({
      data: { type: "samples", buffer: samples.buffer },
    });
  }
}

class FakeAudioContext {
  static instances = [];

  constructor() {
    this.sampleRate = 16_000;
    this.state = "suspended";
    this.destination = {};
    this.audioWorklet = { addModule: vi.fn().mockResolvedValue(undefined) };
    this.source = new FakeAudioNode();
    this.gain = new FakeAudioNode();
    this.gain.gain = { value: 1 };
    this.createMediaStreamSource = vi.fn(() => this.source);
    this.createGain = vi.fn(() => this.gain);
    this.resume = vi.fn(async () => {
      this.state = "running";
    });
    this.close = vi.fn(async () => {
      this.state = "closed";
    });
    FakeAudioContext.instances.push(this);
  }
}

class FakeWebSocket extends EventTarget {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  constructor() {
    super();
    this.readyState = FakeWebSocket.CONNECTING;
    this.bufferedAmount = 0;
    this.binaryType = "blob";
    this.sent = [];
    this.closeCalls = [];
  }

  open() {
    this.readyState = FakeWebSocket.OPEN;
    this.dispatchEvent(new Event("open"));
  }

  send(value) {
    this.sent.push(value);
  }

  message(payload) {
    const event = new Event("message");
    Object.defineProperty(event, "data", { value: JSON.stringify(payload) });
    this.dispatchEvent(event);
  }

  close(code) {
    this.closeCalls.push(code);
    this.readyState = FakeWebSocket.CLOSING;
  }
}

function installLiveBrowser() {
  const track = { stop: vi.fn() };
  const stream = { getTracks: vi.fn(() => [track]) };
  const getUserMedia = vi.fn().mockResolvedValue(stream);
  vi.stubGlobal("navigator", { mediaDevices: { getUserMedia } });
  vi.stubGlobal("AudioContext", FakeAudioContext);
  vi.stubGlobal("AudioWorkletNode", FakeAudioWorkletNode);
  vi.stubGlobal("WebSocket", FakeWebSocket);
  return { getUserMedia, stream, track };
}

function createOpeningSocketFactory() {
  const socket = new FakeWebSocket();
  const factory = vi.fn(() => {
    queueMicrotask(() => socket.open());
    return socket;
  });
  return { factory, socket };
}

async function startSession(callbacks = {}) {
  const { factory, socket } = createOpeningSocketFactory();
  const session = new LiveAnalysisSession({ ...callbacks, webSocketFactory: factory });
  await session.start(`  ${CONTACT_ID}  `);
  return { factory, session, socket, worklet: FakeAudioWorkletNode.instances[0] };
}

beforeEach(() => {
  FakeAudioContext.instances = [];
  FakeAudioWorkletNode.instances = [];
});

describe("LiveAnalysisSession", () => {
  it("sends start first, streams little-endian PCM chunks, and handles progressive/final results", async () => {
    const { track } = installLiveBrowser();
    const onError = vi.fn();
    const onPrediction = vi.fn();
    const onState = vi.fn();
    const { factory, session, socket, worklet } = await startSession({
      onError,
      onPrediction,
      onState,
    });

    expect(factory.mock.calls[0][0]).toMatch(/^ws:\/\/.*\/api\/ws\/analyze$/);
    expect(socket.binaryType).toBe("arraybuffer");
    expect(JSON.parse(socket.sent[0])).toEqual({
      type: "start",
      encoding: "pcm_f32le",
      sample_rate: 16_000,
      channels: 1,
      contact_id: CONTACT_ID,
      persistence_mode: "none",
    });
    expect(FakeAudioContext.instances[0].audioWorklet.addModule).toHaveBeenCalledWith(
      "/pcm-capture-worklet.js",
    );

    const samples = new Float32Array(4_000);
    samples.set([1.5, -1.5, 0.25]);
    worklet.emitSamples(samples);

    expect(socket.sent).toHaveLength(2);
    expect(socket.sent[1]).toBeInstanceOf(ArrayBuffer);
    expect(socket.sent[1].byteLength).toBe(4_000 * 4);
    const pcm = new DataView(socket.sent[1]);
    expect(pcm.getFloat32(0, true)).toBe(1);
    expect(pcm.getFloat32(4, true)).toBe(-1);
    expect(pcm.getFloat32(8, true)).toBeCloseTo(0.25);
    expect(samples.every((sample) => sample === 0)).toBe(true);

    socket.message(prediction(1, false));
    expect(onPrediction).toHaveBeenLastCalledWith(prediction(1, false));

    await Promise.all([session.finish(), session.finish()]);
    const endFrames = socket.sent.filter(
      (value) => typeof value === "string" && value === JSON.stringify({ type: "end" }),
    );
    expect(endFrames).toHaveLength(1);
    expect(track.stop).toHaveBeenCalledTimes(1);
    expect(FakeAudioContext.instances[0].close).toHaveBeenCalledTimes(1);

    socket.message(prediction(2, true));
    expect(onPrediction).toHaveBeenCalledTimes(2);
    expect(onPrediction).toHaveBeenLastCalledWith(prediction(2, true));
    expect(onState).toHaveBeenLastCalledWith("complete");
    expect(onError).not.toHaveBeenCalled();
  });

  it("sends consent-gated audio retention fields in the start frame", async () => {
    installLiveBrowser();
    const { session, socket } = await startSession({
      persistenceMode: "result_and_audio",
      consentReference: "  approval-482  ",
    });

    expect(JSON.parse(socket.sent[0])).toMatchObject({
      type: "start",
      persistence_mode: "result_and_audio",
      consent_reference: "approval-482",
    });

    await session.cancel();
  });

  it("consumes started and storage control events and merges progress into a prediction", async () => {
    installLiveBrowser();
    const onError = vi.fn();
    const onPrediction = vi.fn();
    const onStorage = vi.fn();
    const { session, socket } = await startSession({
      onError,
      onPrediction,
      onStorage,
      persistenceMode: "result_and_audio",
      consentReference: "approval-482",
    });
    const analysisId = "d18f3374-ee30-4cc9-863a-6574f0482e4d";
    const started = persistenceReceipt();
    const progress = persistenceReceipt({
      chunks_received: 3,
      chunks_stored: 2,
      segments_stored: 1,
      bytes_stored: 24_000,
    });

    socket.message({
      type: "started",
      contact_id: CONTACT_ID,
      analysis_id: analysisId,
      persistence: started,
    });
    socket.message({
      type: "storage",
      analysis_id: analysisId,
      persistence: progress,
    });
    socket.message(prediction(1, false));

    expect(onStorage).toHaveBeenNthCalledWith(1, {
      type: "started",
      contact_id: CONTACT_ID,
      analysis_id: analysisId,
      persistence: started,
    });
    expect(onStorage).toHaveBeenNthCalledWith(2, {
      type: "storage",
      analysis_id: analysisId,
      persistence: progress,
    });
    expect(onPrediction).toHaveBeenCalledWith({
      ...prediction(1, false),
      analysis_id: analysisId,
      persistence: progress,
    });
    expect(onError).not.toHaveBeenCalled();

    await session.cancel();
  });

  it("fails closed on socket backpressure without sending or retaining the PCM chunk", async () => {
    const { track } = installLiveBrowser();
    const onError = vi.fn();
    const onState = vi.fn();
    const { session, socket, worklet } = await startSession({ onError, onState });
    socket.bufferedAmount = 512 * 1024;
    const samples = new Float32Array(4_000).fill(0.5);

    worklet.emitSamples(samples);

    await vi.waitFor(() => {
      expect(onError).toHaveBeenCalledWith(
        expect.objectContaining({ code: "STREAM_BACKPRESSURE" }),
      );
    });
    expect(socket.sent).toHaveLength(1);
    expect(session.pending).toEqual([]);
    expect(session.pendingSamples).toBe(0);
    expect(samples.every((sample) => sample === 0)).toBe(true);
    expect(track.stop).toHaveBeenCalledTimes(1);
    expect(socket.closeCalls).toEqual([1000]);
    expect(onState).toHaveBeenLastCalledWith("error");
  });

  it("maps a structured service error and releases microphone/socket resources", async () => {
    const { track } = installLiveBrowser();
    const onError = vi.fn();
    const { socket } = await startSession({ onError });

    socket.message({
      type: "error",
      error: {
        code: "SERVICE_BUSY",
        message: "raw service message",
        request_id: "request-9",
      },
    });

    await vi.waitFor(() => {
      expect(onError).toHaveBeenCalledWith(
        expect.objectContaining({
          code: "SERVICE_BUSY",
          message: "The analyzer is busy. Please start a new live session shortly.",
          requestId: "request-9",
        }),
      );
    });
    expect(track.stop).toHaveBeenCalledTimes(1);
    expect(FakeAudioContext.instances[0].source.disconnect).toHaveBeenCalled();
    expect(FakeAudioWorkletNode.instances[0].disconnect).toHaveBeenCalled();
    expect(FakeAudioContext.instances[0].gain.disconnect).toHaveBeenCalled();
    expect(FakeAudioContext.instances[0].close).toHaveBeenCalledTimes(1);
    expect(socket.closeCalls).toEqual([1000]);
  });

  it("fails closed on malformed service errors instead of exposing object fields", async () => {
    installLiveBrowser();
    const onError = vi.fn();
    const { socket } = await startSession({ onError });

    socket.message({
      type: "error",
      error: {
        code: "SERVICE_BUSY",
        message: "retry",
        request_id: { unsafe: "object" },
      },
    });

    await vi.waitFor(() => {
      expect(onError).toHaveBeenCalledWith(
        expect.objectContaining({ code: "INVALID_STREAM_RESPONSE" }),
      );
    });
  });

  it("does not resurrect predictions received after cancellation", async () => {
    installLiveBrowser();
    const onPrediction = vi.fn();
    const { session, socket } = await startSession({ onPrediction });

    await session.cancel();
    socket.message(prediction(1, false));

    expect(onPrediction).not.toHaveBeenCalled();
  });

  it("releases a microphone granted after live permission was cancelled", async () => {
    let resolvePermission;
    const track = { stop: vi.fn() };
    const getUserMedia = vi.fn(
      () =>
        new Promise((resolve) => {
          resolvePermission = resolve;
        }),
    );
    vi.stubGlobal("navigator", { mediaDevices: { getUserMedia } });
    vi.stubGlobal("AudioContext", FakeAudioContext);
    vi.stubGlobal("AudioWorkletNode", FakeAudioWorkletNode);
    vi.stubGlobal("WebSocket", FakeWebSocket);
    const session = new LiveAnalysisSession();

    const starting = session.start(CONTACT_ID);
    await session.cancel();
    resolvePermission({ getTracks: () => [track] });

    await expect(starting).rejects.toMatchObject({ code: "LIVE_CANCELLED" });
    expect(track.stop).toHaveBeenCalledTimes(1);
    expect(FakeAudioContext.instances[0].close).toHaveBeenCalledTimes(1);
  });

  it("builds secure and insecure same-origin WebSocket URLs", () => {
    expect(analysisWebSocketUrl({ protocol: "https:", host: "voice.example" })).toBe(
      "wss://voice.example/api/ws/analyze",
    );
    expect(analysisWebSocketUrl({ protocol: "http:", host: "localhost:3000" })).toBe(
      "ws://localhost:3000/api/ws/analyze",
    );
  });
});
