export const MAX_CAPTURE_SECONDS = 29;

const RECORDING_TYPES = [
  "audio/webm;codecs=opus",
  "audio/ogg;codecs=opus",
  "audio/mp4",
  "audio/webm",
];

const MICROPHONE_CONSTRAINTS = {
  audio: {
    autoGainControl: { ideal: false },
    channelCount: { ideal: 1 },
    echoCancellation: { ideal: true },
    noiseSuppression: { ideal: true },
  },
  video: false,
};

export class MicrophoneCaptureError extends Error {
  constructor(message, code = "MICROPHONE_ERROR") {
    super(message);
    this.name = "MicrophoneCaptureError";
    this.code = code;
  }
}

export function microphoneError(error) {
  if (error instanceof MicrophoneCaptureError) return error;
  const byName = {
    AbortError: ["The browser could not start microphone capture. Please retry.", "MIC_ABORTED"],
    NotAllowedError: [
      "Microphone permission was denied. Allow access in the browser and try again.",
      "MIC_PERMISSION_DENIED",
    ],
    NotFoundError: ["No microphone was found on this device.", "MIC_NOT_FOUND"],
    NotReadableError: [
      "The microphone is busy or unavailable to the browser.",
      "MIC_UNAVAILABLE",
    ],
    SecurityError: [
      "Microphone access requires HTTPS, except when using localhost.",
      "MIC_INSECURE_CONTEXT",
    ],
  };
  const [message, code] = byName[error?.name] || [
    "The microphone could not be started. Check browser permissions and try again.",
    "MICROPHONE_ERROR",
  ];
  return new MicrophoneCaptureError(message, code);
}

export function hasMicrophoneSupport() {
  return Boolean(navigator.mediaDevices?.getUserMedia);
}

export function hasRecordingSupport() {
  return hasMicrophoneSupport() && typeof MediaRecorder !== "undefined";
}

export function hasLiveCaptureSupport() {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  return Boolean(
    hasMicrophoneSupport() &&
      AudioContextClass &&
      typeof AudioWorkletNode !== "undefined",
  );
}

export async function requestMicrophone() {
  if (window.isSecureContext === false) {
    throw new MicrophoneCaptureError(
      "Microphone access requires HTTPS, except when using localhost.",
      "MIC_INSECURE_CONTEXT",
    );
  }
  if (!hasMicrophoneSupport()) {
    throw new MicrophoneCaptureError(
      "This browser does not support microphone capture.",
      "MIC_UNSUPPORTED",
    );
  }
  try {
    return await navigator.mediaDevices.getUserMedia(MICROPHONE_CONSTRAINTS);
  } catch (error) {
    throw microphoneError(error);
  }
}

function preferredRecordingType() {
  if (typeof MediaRecorder === "undefined") return "";
  if (typeof MediaRecorder.isTypeSupported !== "function") return "";
  return RECORDING_TYPES.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

function extensionFor(type) {
  if (type.includes("mp4")) return "m4a";
  if (type.includes("ogg")) return "ogg";
  return "webm";
}

function stopTracks(stream) {
  for (const track of stream?.getTracks?.() || []) track.stop();
}

export class BrowserRecorder {
  constructor({ onUnexpectedStop } = {}) {
    this.onUnexpectedStop = onUnexpectedStop || (() => {});
    this.stream = null;
    this.recorder = null;
    this.chunks = [];
    this.stopPromise = null;
    this.resolveStop = null;
    this.rejectStop = null;
    this.terminalError = null;
    this.stopRequested = false;
    this.finished = false;
    this.terminationNotified = false;
    this.cancelled = false;
  }

  releaseStream() {
    stopTracks(this.stream);
    this.stream = null;
  }

  notifyUnexpectedStop(error) {
    if (this.stopRequested || this.cancelled || this.terminationNotified) return;
    this.terminationNotified = true;
    try {
      this.onUnexpectedStop(error);
    } catch {
      // Resource cleanup and deferred settlement must not depend on UI callbacks.
    }
  }

  completeStop() {
    if (this.finished) return;
    this.finished = true;
    this.releaseStream();

    if (!this.stopRequested && !this.cancelled && !this.terminalError) {
      this.terminalError = new MicrophoneCaptureError(
        "Microphone recording stopped unexpectedly. Please retry.",
        "RECORDING_INTERRUPTED",
      );
      this.notifyUnexpectedStop(this.terminalError);
    }

    if (this.cancelled) {
      this.chunks = [];
      this.rejectStop?.(
        new MicrophoneCaptureError("Recording cancelled.", "RECORDING_CANCELLED"),
      );
      return;
    }
    if (this.terminalError) {
      this.chunks = [];
      this.rejectStop?.(this.terminalError);
      return;
    }

    const type = this.recorder?.mimeType || this.chunks[0]?.type || "audio/webm";
    const blob = new Blob(this.chunks, { type });
    this.chunks = [];
    if (!blob.size) {
      this.rejectStop?.(
        new MicrophoneCaptureError(
          "The browser did not capture any audio. Please retry.",
          "EMPTY_RECORDING",
        ),
      );
      return;
    }
    const filename = `microphone-${new Date().toISOString().replace(/[:.]/g, "-")}.${extensionFor(type)}`;
    this.resolveStop?.(new File([blob], filename, { type }));
  }

  async start() {
    if (!hasRecordingSupport()) {
      throw new MicrophoneCaptureError(
        "This browser cannot create an audio recording.",
        "RECORDING_UNSUPPORTED",
      );
    }

    this.stream = await requestMicrophone();
    if (this.cancelled) {
      this.releaseStream();
      throw new MicrophoneCaptureError("Recording cancelled.", "RECORDING_CANCELLED");
    }
    const requestedType = preferredRecordingType();
    try {
      this.recorder = requestedType
        ? new MediaRecorder(this.stream, { mimeType: requestedType })
        : new MediaRecorder(this.stream);
    } catch (error) {
      this.releaseStream();
      throw microphoneError(error);
    }

    this.stopPromise = new Promise((resolve, reject) => {
      this.resolveStop = resolve;
      this.rejectStop = reject;
    });
    // A recorder can fail before the UI asks it to stop. Keep that deferred
    // rejection observed while still returning the same promise from stop().
    void this.stopPromise.catch(() => {});
    this.recorder.addEventListener("dataavailable", (event) => {
      if (!this.cancelled && !this.terminalError && event.data?.size) {
        this.chunks.push(event.data);
      }
    });
    this.recorder.addEventListener("error", (event) => {
      this.terminalError = microphoneError(event.error);
      this.chunks = [];
      this.releaseStream();
      this.notifyUnexpectedStop(this.terminalError);
    });
    this.recorder.addEventListener("stop", () => this.completeStop(), { once: true });
    try {
      this.recorder.start(500);
    } catch (error) {
      this.releaseStream();
      this.stopPromise = null;
      this.resolveStop = null;
      this.rejectStop = null;
      throw microphoneError(error);
    }
    return { mimeType: this.recorder.mimeType || requestedType || "audio/webm" };
  }

  stop() {
    if (!this.recorder || !this.stopPromise) {
      return Promise.reject(
        new MicrophoneCaptureError("No recording is active.", "RECORDING_NOT_ACTIVE"),
      );
    }
    if (this.stopRequested) return this.stopPromise;
    this.stopRequested = true;
    if (this.recorder.state !== "inactive") this.recorder.stop();
    return this.stopPromise;
  }

  cancel() {
    this.cancelled = true;
    this.chunks = [];
    if (this.recorder && this.recorder.state !== "inactive") this.recorder.stop();
    this.releaseStream();
  }
}

export function formatCaptureTime(seconds) {
  const bounded = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(bounded / 60)).padStart(2, "0")}:${String(bounded % 60).padStart(2, "0")}`;
}
