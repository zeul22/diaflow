import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { StrictMode } from "react";

import App from "./App.jsx";
import { MAX_AUDIO_BYTES } from "./api/analyze.js";
import { LiveAnalysisSession } from "./api/streamAnalysis.js";

function response(payload, { ok = true, status = 200, requestId = "request-123" } = {}) {
  return {
    ok,
    status,
    headers: new Headers({ "X-Request-ID": requestId }),
    json: vi.fn().mockResolvedValue(payload),
  };
}

const successfulResult = {
  contact_id: "123e4567-e89b-12d3-a456-426614174000",
  gender: { prediction: "unknown", confidence: 0 },
  age_bracket: { prediction: "31-45", confidence: 0.4095 },
  processing_ms: 302,
  audio_quality: "degraded",
};

function renderApp() {
  return render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}

function mockAnalyzer(analyzerPromise = Promise.resolve(response(successfulResult))) {
  const fetchMock = vi.fn((url) => {
    if (url === "/api/readyz") {
      return Promise.resolve(response({ status: "ready" }));
    }
    return analyzerPromise;
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function analyzerCalls(fetchMock) {
  return fetchMock.mock.calls.filter(([url]) => url === "/api/analyze");
}

function installRecorderBrowser({ getUserMediaImpl } = {}) {
  const track = { stop: vi.fn() };
  const stream = { getTracks: () => [track] };
  const getUserMedia = vi.fn(
    getUserMediaImpl || (() => Promise.resolve(stream)),
  );
  const instances = [];
  vi.stubGlobal("navigator", {
    clipboard: { writeText: vi.fn() },
    mediaDevices: { getUserMedia },
  });

  class TestMediaRecorder extends EventTarget {
    static isTypeSupported(type) {
      return type === "audio/webm;codecs=opus";
    }

    constructor() {
      super();
      this.mimeType = "audio/webm;codecs=opus";
      this.state = "inactive";
      instances.push(this);
    }

    start() {
      this.state = "recording";
    }

    stop() {
      this.state = "inactive";
      const dataEvent = new Event("dataavailable");
      Object.defineProperty(dataEvent, "data", {
        value: new Blob(["recorded caller audio"], { type: this.mimeType }),
      });
      this.dispatchEvent(dataEvent);
      this.dispatchEvent(new Event("stop"));
    }
  }
  vi.stubGlobal("MediaRecorder", TestMediaRecorder);
  return { getUserMedia, instances, stream, track };
}

function installLiveSupport() {
  vi.stubGlobal("navigator", {
    clipboard: { writeText: vi.fn() },
    mediaDevices: { getUserMedia: vi.fn() },
  });
  vi.stubGlobal("AudioContext", class TestAudioContext {});
  vi.stubGlobal("AudioWorkletNode", class TestAudioWorkletNode {});
}

describe("Audio analyzer", () => {
  it("uploads M4A as multipart and renders an abstaining result", async () => {
    const user = userEvent.setup();
    const fetchMock = mockAnalyzer();
    renderApp();

    const file = new File(["m4a audio"], "trial.m4a", { type: "audio/mp4" });
    await user.upload(screen.getByLabelText(/browse files/i), file);
    await user.type(
      screen.getByLabelText(/contact id/i),
      "123e4567-e89b-12d3-a456-426614174000",
    );
    await user.click(screen.getByRole("button", { name: /analyze audio/i }));

    expect(await screen.findByRole("heading", { name: /contact attributes/i })).toBeVisible();
    expect(screen.getByText("Audio quality is degraded")).toBeVisible();
    expect(screen.getByText("31-45")).toBeVisible();
    expect(screen.getByText("41%")).toBeVisible();
    expect(screen.getByText("302 ms")).toBeVisible();
    expect(screen.getByText("Model abstained")).toBeVisible();

    const calls = analyzerCalls(fetchMock);
    expect(calls).toHaveLength(1);
    const [url, options] = calls[0];
    expect(url).toBe("/api/analyze");
    expect(options.method).toBe("POST");
    expect(options.body).toBeInstanceOf(FormData);
    const uploadedAudio = options.body.get("audio");
    expect(uploadedAudio).toBeInstanceOf(File);
    expect(uploadedAudio.name).toBe("trial.m4a");
    expect(uploadedAudio.type).toBe("audio/mp4");
    expect(uploadedAudio.size).toBe(file.size);
    expect(options.body.get("contact_id")).toBe(
      "123e4567-e89b-12d3-a456-426614174000",
    );
    expect(options.headers).not.toHaveProperty("Content-Type");

    await user.click(screen.getByRole("button", { name: /analyze another/i }));
    expect(screen.getByLabelText(/contact id/i)).toHaveValue("");
  });

  it("rejects oversized audio before making a request", async () => {
    const user = userEvent.setup();
    const fetchMock = mockAnalyzer();
    renderApp();

    const file = new File(["audio"], "too-large.m4a", { type: "audio/mp4" });
    Object.defineProperty(file, "size", { value: MAX_AUDIO_BYTES + 1 });
    await user.upload(screen.getByLabelText(/browse files/i), file);

    expect(screen.getByRole("alert")).toHaveTextContent("smaller than 12 MB");
    expect(analyzerCalls(fetchMock)).toHaveLength(0);
  });

  it("shows the structured service error and keeps the file selected for retry", async () => {
    const user = userEvent.setup();
    mockAnalyzer(
      Promise.resolve(
      response(
        {
          error: {
            code: "SERVICE_BUSY",
            message: "Inference capacity is temporarily full",
            request_id: "busy-request-9",
          },
        },
        { ok: false, status: 503, requestId: "busy-request-9" },
      ),
      ),
    );
    renderApp();

    const file = new File(["audio"], "caller.wav", { type: "audio/wav" });
    await user.upload(screen.getByLabelText(/browse files/i), file);
    await user.click(screen.getByRole("button", { name: /analyze audio/i }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("analyzer is busy");
    expect(alert).toHaveTextContent("busy-request-9");
    expect(screen.getByText("caller.wav")).toBeVisible();
    expect(screen.getByRole("button", { name: /analyze audio/i })).toBeEnabled();
  });

  it("prevents duplicate submissions while analysis is active", async () => {
    const user = userEvent.setup();
    let resolveRequest;
    const pending = new Promise((resolve) => {
      resolveRequest = resolve;
    });
    const fetchMock = mockAnalyzer(pending);
    renderApp();

    const file = new File(["audio"], "caller.wav", { type: "audio/wav" });
    await user.upload(screen.getByLabelText(/browse files/i), file);
    const submit = screen.getByRole("button", { name: /analyze audio/i });
    await user.click(submit);

    expect(screen.getByRole("button", { name: /analyzing securely/i })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: /analyzing securely/i }));
    expect(analyzerCalls(fetchMock)).toHaveLength(1);

    resolveRequest(response(successfulResult));
    await waitFor(() => {
      expect(screen.getByRole("heading", { name: /contact attributes/i })).toBeVisible();
    });
  });

  it("revokes the browser preview URL when audio is removed", async () => {
    const user = userEvent.setup();
    mockAnalyzer();
    renderApp();

    const file = new File(["audio"], "caller.wav", { type: "audio/wav" });
    await user.upload(screen.getByLabelText(/browse files/i), file);
    expect(URL.createObjectURL).toHaveBeenCalledWith(file);
    await user.click(screen.getByRole("button", { name: /remove/i }));
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:audio-preview");
    expect(URL.revokeObjectURL).toHaveBeenCalledTimes(URL.createObjectURL.mock.calls.length);
  });

  it("rejects malformed success payloads", async () => {
    const user = userEvent.setup();
    mockAnalyzer(
      Promise.resolve(
        response({
          ...successfulResult,
          gender: { prediction: "male", confidence: 4.2 },
        }),
      ),
    );
    renderApp();

    const file = new File(["audio"], "caller.wav", { type: "audio/wav" });
    await user.upload(screen.getByLabelText(/browse files/i), file);
    await user.click(screen.getByRole("button", { name: /analyze audio/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent("unexpected response");
  });

  it("records microphone audio and reuses the REST analysis flow", async () => {
    const user = userEvent.setup();
    const fetchMock = mockAnalyzer();
    const { getUserMedia, track } = installRecorderBrowser();
    renderApp();

    await user.click(screen.getByRole("radio", { name: /^record/i }));
    expect(screen.getByRole("heading", { name: "Record caller audio" })).toBeVisible();
    expect(getUserMedia).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: /start recording/i }));
    expect(await screen.findByText("Recording in progress")).toBeVisible();
    expect(getUserMedia).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: /stop recording/i }));
    expect(await screen.findByRole("button", { name: /analyze recording/i })).toBeEnabled();
    expect(screen.getByText(/microphone-.*\.webm/i)).toBeVisible();
    expect(track.stop).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: /analyze recording/i }));
    expect(await screen.findByRole("heading", { name: /contact attributes/i })).toBeVisible();
    const calls = analyzerCalls(fetchMock);
    expect(calls).toHaveLength(1);
    expect(calls[0][1].body.get("audio").type).toBe("audio/webm;codecs=opus");
  });

  it("can cancel an ignored microphone permission request without later restarting", async () => {
    const user = userEvent.setup();
    mockAnalyzer();
    let resolvePermission;
    const pendingPermission = new Promise((resolve) => {
      resolvePermission = resolve;
    });
    const { stream, track } = installRecorderBrowser({
      getUserMediaImpl: () => pendingPermission,
    });
    renderApp();

    await user.click(screen.getByRole("radio", { name: /^record/i }));
    await user.click(screen.getByRole("button", { name: /start recording/i }));
    expect(screen.getByRole("button", { name: /cancel request/i })).toBeEnabled();
    await user.click(screen.getByRole("button", { name: /cancel request/i }));
    expect(screen.getByRole("button", { name: /start recording/i })).toBeEnabled();

    resolvePermission(stream);
    await waitFor(() => expect(track.stop).toHaveBeenCalledTimes(1));
    expect(screen.getByRole("button", { name: /start recording/i })).toBeEnabled();
    expect(screen.queryByText("Recording in progress")).not.toBeInTheDocument();
  });

  it("exits recording and releases the microphone after a recorder runtime error", async () => {
    const user = userEvent.setup();
    mockAnalyzer();
    const { instances, track } = installRecorderBrowser();
    renderApp();

    await user.click(screen.getByRole("radio", { name: /^record/i }));
    await user.click(screen.getByRole("button", { name: /start recording/i }));
    expect(await screen.findByText("Recording in progress")).toBeVisible();

    const failure = new Error("device disappeared");
    failure.name = "NotReadableError";
    const errorEvent = new Event("error");
    Object.defineProperty(errorEvent, "error", { value: failure });
    instances[0].state = "inactive";
    instances[0].dispatchEvent(errorEvent);
    instances[0].dispatchEvent(new Event("stop"));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "microphone is busy or unavailable",
    );
    expect(screen.getByRole("button", { name: /start recording/i })).toBeEnabled();
    expect(track.stop).toHaveBeenCalledTimes(1);
  });

  it("shows provisional live updates and settles the final WebSocket result", async () => {
    const user = userEvent.setup();
    mockAnalyzer();
    installLiveSupport();
    const progressive = {
      ...successfulResult,
      type: "prediction",
      sequence: 1,
      is_final: false,
    };
    const final = { ...progressive, sequence: 2, is_final: true };
    vi.spyOn(LiveAnalysisSession.prototype, "start").mockImplementation(async function () {
      this.onState("streaming");
      this.onStats({ bytes: 16000, chunks: 2, elapsed: 1.5, level: 0.45, sampleRate: 48000 });
      this.onPrediction(progressive);
    });
    vi.spyOn(LiveAnalysisSession.prototype, "finish").mockImplementation(async function () {
      this.onPrediction(final);
      this.onState("complete");
    });

    renderApp();
    await user.click(screen.getByRole("radio", { name: /^live/i }));
    await user.click(screen.getByRole("button", { name: /start live analysis/i }));

    expect(await screen.findByText("Estimate may change")).toBeVisible();
    expect(screen.getByText(/live · update 1/i)).toBeVisible();
    expect(screen.getByText(/2 raw pcm chunks/i)).toBeVisible();

    await user.click(screen.getByRole("button", { name: /stop & finalize/i }));
    expect(await screen.findByText("Final result")).toBeVisible();
    expect(screen.getByRole("button", { name: /analyze another/i })).toBeEnabled();
  });

  it("does not let a slow cancelled session clobber a newly started live session", async () => {
    const user = userEvent.setup();
    mockAnalyzer();
    installLiveSupport();
    let releaseFirstCancel;
    const firstCancel = new Promise((resolve) => {
      releaseFirstCancel = resolve;
    });
    const start = vi
      .spyOn(LiveAnalysisSession.prototype, "start")
      .mockImplementation(async function () {
        this.onState("streaming");
      });
    vi.spyOn(LiveAnalysisSession.prototype, "cancel")
      .mockImplementationOnce(() => firstCancel)
      .mockResolvedValue(undefined);

    renderApp();
    await user.click(screen.getByRole("radio", { name: /^live/i }));
    await user.click(screen.getByRole("button", { name: /start live analysis/i }));
    await user.click(screen.getByRole("button", { name: /cancel stream/i }));
    await user.click(screen.getByRole("button", { name: /start live analysis/i }));

    expect(start).toHaveBeenCalledTimes(2);
    expect(screen.getByRole("button", { name: /stop & finalize/i })).toBeEnabled();
    releaseFirstCancel();
    await Promise.resolve();
    expect(screen.getByRole("button", { name: /stop & finalize/i })).toBeEnabled();
  });
});
