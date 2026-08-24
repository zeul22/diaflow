import { describe, expect, it, vi } from "vitest";

import { BrowserRecorder, MicrophoneCaptureError } from "./microphone.js";

function installMicrophone() {
  const tracks = [{ stop: vi.fn() }, { stop: vi.fn() }];
  const stream = { getTracks: vi.fn(() => tracks) };
  const getUserMedia = vi.fn().mockResolvedValue(stream);
  vi.stubGlobal("navigator", { mediaDevices: { getUserMedia } });
  return { getUserMedia, stream, tracks };
}

function installMediaRecorder({
  chunks = ["captured audio"],
  supported = (type) => type === "audio/webm;codecs=opus",
} = {}) {
  const instances = [];

  class FakeMediaRecorder extends EventTarget {
    static isTypeSupported = vi.fn(supported);

    constructor(stream, options = {}) {
      super();
      this.stream = stream;
      this.options = options;
      this.mimeType = options.mimeType || "audio/webm";
      this.state = "inactive";
      this.start = vi.fn((timeslice) => {
        this.timeslice = timeslice;
        this.state = "recording";
      });
      this.stop = vi.fn(() => {
        this.state = "inactive";
        for (const value of chunks) {
          const event = new Event("dataavailable");
          Object.defineProperty(event, "data", {
            value: new Blob([value], { type: this.mimeType }),
          });
          this.dispatchEvent(event);
        }
        this.dispatchEvent(new Event("stop"));
      });
      instances.push(this);
    }
  }

  vi.stubGlobal("MediaRecorder", FakeMediaRecorder);
  return { FakeMediaRecorder, instances };
}

describe("BrowserRecorder", () => {
  it("chooses the preferred MIME type and returns a matching File", async () => {
    const { getUserMedia, stream, tracks } = installMicrophone();
    const { FakeMediaRecorder, instances } = installMediaRecorder();
    const recorder = new BrowserRecorder();

    await expect(recorder.start()).resolves.toEqual({
      mimeType: "audio/webm;codecs=opus",
    });

    expect(getUserMedia).toHaveBeenCalledWith({
      audio: {
        autoGainControl: { ideal: false },
        channelCount: { ideal: 1 },
        echoCancellation: { ideal: true },
        noiseSuppression: { ideal: true },
      },
      video: false,
    });
    expect(FakeMediaRecorder.isTypeSupported).toHaveBeenCalledWith(
      "audio/webm;codecs=opus",
    );
    expect(instances[0].stream).toBe(stream);
    expect(instances[0].options).toEqual({ mimeType: "audio/webm;codecs=opus" });
    expect(instances[0].start).toHaveBeenCalledWith(500);

    const file = await recorder.stop();

    expect(file).toBeInstanceOf(File);
    expect(file.type).toBe("audio/webm;codecs=opus");
    expect(file.name).toMatch(/^microphone-.*\.webm$/);
    expect(file.size).toBeGreaterThan(0);
    for (const track of tracks) expect(track.stop).toHaveBeenCalledTimes(1);
  });

  it("uses an M4A filename when MP4 is the first supported recording format", async () => {
    installMicrophone();
    installMediaRecorder({ supported: (type) => type === "audio/mp4" });
    const recorder = new BrowserRecorder();

    await recorder.start();
    const file = await recorder.stop();

    expect(file.type).toBe("audio/mp4");
    expect(file.name).toMatch(/\.m4a$/);
  });

  it("discards captured chunks and stops every track when cancelled", async () => {
    const { tracks } = installMicrophone();
    const { instances } = installMediaRecorder();
    const recorder = new BrowserRecorder();

    await recorder.start();
    recorder.cancel();
    recorder.cancel();

    expect(instances[0].stop).toHaveBeenCalledTimes(1);
    expect(recorder.chunks).toEqual([]);
    expect(recorder.stream).toBeNull();
    for (const track of tracks) expect(track.stop).toHaveBeenCalledTimes(1);
    await expect(recorder.stop()).rejects.toMatchObject({ code: "RECORDING_CANCELLED" });
  });

  it("releases a microphone that arrives after a pending permission request is cancelled", async () => {
    let resolvePermission;
    const tracks = [{ stop: vi.fn() }];
    const getUserMedia = vi.fn(
      () =>
        new Promise((resolve) => {
          resolvePermission = resolve;
        }),
    );
    vi.stubGlobal("navigator", { mediaDevices: { getUserMedia } });
    const { instances } = installMediaRecorder();
    const recorder = new BrowserRecorder();

    const starting = recorder.start();
    recorder.cancel();
    resolvePermission({ getTracks: () => tracks });

    await expect(starting).rejects.toMatchObject({ code: "RECORDING_CANCELLED" });
    expect(instances).toHaveLength(0);
    expect(tracks[0].stop).toHaveBeenCalledTimes(1);
  });

  it("observes runtime recorder errors and releases tracks before Stop is pressed", async () => {
    const { tracks } = installMicrophone();
    const { instances } = installMediaRecorder();
    const onUnexpectedStop = vi.fn();
    const recorder = new BrowserRecorder({ onUnexpectedStop });
    await recorder.start();

    const failure = new Error("microphone disconnected");
    failure.name = "NotReadableError";
    const errorEvent = new Event("error");
    Object.defineProperty(errorEvent, "error", { value: failure });
    instances[0].state = "inactive";
    instances[0].dispatchEvent(errorEvent);
    instances[0].dispatchEvent(new Event("stop"));

    expect(onUnexpectedStop).toHaveBeenCalledWith(
      expect.objectContaining({ code: "MIC_UNAVAILABLE" }),
    );
    for (const track of tracks) expect(track.stop).toHaveBeenCalledTimes(1);
    await expect(recorder.stop()).rejects.toMatchObject({ code: "MIC_UNAVAILABLE" });
  });

  it("stops acquired tracks if MediaRecorder construction fails", async () => {
    const { tracks } = installMicrophone();
    class FailingMediaRecorder {
      static isTypeSupported() {
        return true;
      }

      constructor() {
        const error = new Error("device disappeared");
        error.name = "NotReadableError";
        throw error;
      }
    }
    vi.stubGlobal("MediaRecorder", FailingMediaRecorder);
    const recorder = new BrowserRecorder();

    await expect(recorder.start()).rejects.toEqual(
      expect.objectContaining({
        code: "MIC_UNAVAILABLE",
        name: "MicrophoneCaptureError",
      }),
    );
    expect(recorder.stream).toBeNull();
    for (const track of tracks) expect(track.stop).toHaveBeenCalledTimes(1);
  });

  it("rejects stop before start with a typed error", async () => {
    const recorder = new BrowserRecorder();

    await expect(recorder.stop()).rejects.toBeInstanceOf(MicrophoneCaptureError);
    await expect(recorder.stop()).rejects.toMatchObject({ code: "RECORDING_NOT_ACTIVE" });
  });
});
