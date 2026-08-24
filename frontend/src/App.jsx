import { useEffect, useRef, useState } from "react";

import {
  AnalysisApiError,
  analyzeAudio,
  formatBytes,
  validateAudioFile,
} from "./api/analyze.js";
import { LiveAnalysisSession } from "./api/streamAnalysis.js";
import {
  BrowserRecorder,
  MAX_CAPTURE_SECONDS,
  formatCaptureTime,
  hasLiveCaptureSupport,
  hasRecordingSupport,
  microphoneError,
} from "./audio/microphone.js";

const QUALITY_COPY = {
  good: {
    eyebrow: "Clear signal",
    title: "Audio quality is good",
    detail: "The recording passed the service's acoustic quality checks.",
  },
  degraded: {
    eyebrow: "Use with caution",
    title: "Audio quality is degraded",
    detail: "Compression or noise reduced confidence. Try 3–5 seconds of clearer caller-only speech.",
  },
  insufficient: {
    eyebrow: "Model abstained",
    title: "Not enough usable speech",
    detail: "Both attributes were intentionally withheld. Try a longer, clearer recording.",
  },
};

const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

function WaveIcon() {
  return (
    <svg viewBox="0 0 32 32" aria-hidden="true">
      <path d="M3 17h3l2-8 4 15 4-19 4 22 3-14 2 4h4" />
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M5 14.5v3A2.5 2.5 0 0 0 7.5 20h9a2.5 2.5 0 0 0 2.5-2.5v-3" />
    </svg>
  );
}

function MicrophoneIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <rect x="8" y="3" width="8" height="12" rx="4" />
      <path d="M5.5 11.5a6.5 6.5 0 0 0 13 0M12 18v3M8.5 21h7" />
    </svg>
  );
}

function LiveIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="2" />
      <path d="M7.8 7.8a6 6 0 0 0 0 8.4m8.4-8.4a6 6 0 0 1 0 8.4M4.6 4.6a10.5 10.5 0 0 0 0 14.8m14.8-14.8a10.5 10.5 0 0 1 0 14.8" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg viewBox="0 0 20 20" aria-hidden="true">
      <path d="m4 10 4 4 8-9" />
    </svg>
  );
}

function LockIcon() {
  return (
    <svg viewBox="0 0 20 20" aria-hidden="true">
      <rect x="4" y="8" width="12" height="9" rx="2" />
      <path d="M7 8V6a3 3 0 0 1 6 0v2" />
    </svg>
  );
}

function AudioPreview({ file }) {
  const audioRef = useRef(null);

  useEffect(() => {
    if (!audioRef.current || typeof URL.createObjectURL !== "function") return undefined;
    const audioElement = audioRef.current;
    const objectUrl = URL.createObjectURL(file);
    audioElement.src = objectUrl;
    return () => {
      audioElement.removeAttribute("src");
      if (typeof URL.revokeObjectURL === "function") URL.revokeObjectURL(objectUrl);
    };
  }, [file]);

  return (
    <audio ref={audioRef} className="audio-preview" controls preload="metadata">
      Your browser cannot preview this audio format.
    </audio>
  );
}

function FileSummary({ file, onRemove }) {
  return (
    <div className="file-summary">
      <div className="file-summary__topline">
        <div className="file-summary__icon">
          <WaveIcon />
        </div>
        <div className="file-summary__details">
          <strong title={file.name}>{file.name}</strong>
          <span>
            {formatBytes(file.size)} · {file.type || "Audio file"}
          </span>
        </div>
        <button className="text-button" type="button" onClick={onRemove}>
          Remove
        </button>
      </div>
      <AudioPreview file={file} />
    </div>
  );
}

function PredictionCard({ label, prediction }) {
  const unknown = prediction.prediction === "unknown";
  const confidence = Math.round(prediction.confidence * 100);

  return (
    <article className={`prediction-card${unknown ? " prediction-card--unknown" : ""}`}>
      <p className="prediction-card__label">{label}</p>
      <div className="prediction-card__value">
        {unknown ? "Model abstained" : prediction.prediction}
      </div>
      {unknown ? (
        <p className="prediction-card__note">
          The score did not meet the configured confidence threshold.
        </p>
      ) : (
        <>
          <div className="confidence-row">
            <span>Confidence</span>
            <strong>{confidence}%</strong>
          </div>
          <div
            className="confidence-meter"
            role="progressbar"
            aria-label={`${label} confidence`}
            aria-valuemin="0"
            aria-valuemax="100"
            aria-valuenow={confidence}
          >
            <span style={{ "--confidence": `${confidence}%` }} />
          </div>
        </>
      )}
    </article>
  );
}

function ResultsPanel({ result, headingRef, onReset, streamStats }) {
  const [copied, setCopied] = useState(false);
  const quality = QUALITY_COPY[result.audio_quality] || QUALITY_COPY.degraded;
  const isLive = result.type === "prediction";
  const isFinal = !isLive || result.is_final;

  async function copyContactId() {
    if (!navigator.clipboard) return;
    try {
      await navigator.clipboard.writeText(result.contact_id);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  }

  return (
    <section className="results-panel" aria-labelledby="results-title">
      <div className="results-panel__heading">
        <div>
          <p className="section-kicker">
            {isLive ? (isFinal ? "Final live result" : "Provisional live estimate") : "Analysis complete"}
          </p>
          <h2 id="results-title" ref={headingRef} tabIndex="-1">
            Contact attributes
          </h2>
        </div>
        {isFinal ? (
          <button className="secondary-button" type="button" onClick={onReset}>
            Analyze another
          </button>
        ) : (
          <span className="live-badge"><span /> Live · update {result.sequence}</span>
        )}
      </div>

      {isLive ? (
        <div className={`stream-result-note${isFinal ? " stream-result-note--final" : ""}`}>
          <strong>{isFinal ? "Final result" : "Estimate may change"}</strong>
          <span>
            {formatCaptureTime(streamStats?.elapsed || 0)} captured · {streamStats?.chunks || 0} raw PCM chunks
          </span>
        </div>
      ) : null}

      <div className={`quality-banner quality-banner--${result.audio_quality}`}>
        <div className="quality-banner__icon">
          <CheckIcon />
        </div>
        <div>
          <span>{quality.eyebrow}</span>
          <strong>{quality.title}</strong>
          <p>{quality.detail}</p>
        </div>
      </div>

      <div className="prediction-grid">
        <PredictionCard label="Perceived voice presentation" prediction={result.gender} />
        <PredictionCard label="Estimated age bracket" prediction={result.age_bracket} />
      </div>

      <dl className="result-metadata">
        <div>
          <dt>Model processing</dt>
          <dd>{result.processing_ms} ms</dd>
        </div>
        <div className="result-metadata__contact">
          <dt>Contact ID</dt>
          <dd title={result.contact_id}>{result.contact_id}</dd>
          <button type="button" onClick={copyContactId} disabled={!navigator.clipboard}>
            {copied ? "Copied" : "Copy"}
          </button>
        </div>
      </dl>

      <div className="responsible-use">
        <strong>Interpret carefully.</strong> Voice presentation is not gender identity, and
        these confidence scores are not yet calibrated for logistics calls. Do not use the
        result for consequential decisions.
      </div>
    </section>
  );
}

function EmptyResults() {
  return (
    <aside className="empty-results" aria-label="How analysis works">
      <p className="section-kicker">What happens next</p>
      <h2>One short sample. Clear, bounded output.</h2>
      <ol className="process-list">
        <li>
          <span>01</span>
          <div>
            <strong>Quality check</strong>
            <p>Detects insufficient, noisy, clipped, or compressed audio.</p>
          </div>
        </li>
        <li>
          <span>02</span>
          <div>
            <strong>Shared voice embedding</strong>
            <p>Runs one CPU-efficient encoder over caller-only speech.</p>
          </div>
        </li>
        <li>
          <span>03</span>
          <div>
            <strong>Confidence-aware result</strong>
            <p>Returns an estimate—or abstains when the signal is uncertain.</p>
          </div>
        </li>
      </ol>
      <div className="privacy-card">
        <LockIcon />
        <div>
          <strong>Request-scoped by design</strong>
          <p>No browser persistence, analytics, or application-level audio storage.</p>
        </div>
      </div>
    </aside>
  );
}

function ServiceStatus() {
  const [status, setStatus] = useState("checking");

  useEffect(() => {
    const controller = new AbortController();

    async function checkReadiness() {
      try {
        const response = await fetch("/api/readyz", {
          cache: "no-store",
          headers: { Accept: "application/json" },
          signal: controller.signal,
        });
        setStatus(response.ok ? "ready" : "starting");
      } catch (readinessError) {
        if (readinessError?.name !== "AbortError") setStatus("offline");
      }
    }

    checkReadiness();
    const interval = window.setInterval(checkReadiness, 15_000);
    return () => {
      controller.abort();
      window.clearInterval(interval);
    };
  }, []);

  const copy = {
    checking: "Checking analyzer",
    offline: "Analyzer offline",
    ready: "Analyzer ready",
    starting: "Analyzer starting",
  }[status];

  return (
    <div className={`header-status header-status--${status}`} role="status">
      <span /> {copy}
    </div>
  );
}

const SOURCE_OPTIONS = [
  { value: "upload", label: "Upload", detail: "Existing file", icon: UploadIcon },
  { value: "record", label: "Record", detail: "Capture & preview", icon: MicrophoneIcon },
  { value: "live", label: "Live", detail: "Progressive results", icon: LiveIcon },
];

function SourceSelector({ value, disabled, onChange }) {
  return (
    <fieldset className="source-selector" disabled={disabled}>
      <legend>Audio source</legend>
      {SOURCE_OPTIONS.map((option) => {
        const Icon = option.icon;
        return (
          <label key={option.value} className={value === option.value ? "is-selected" : ""}>
            <input
              type="radio"
              name="audio-source"
              value={option.value}
              checked={value === option.value}
              onChange={() => onChange(option.value)}
            />
            <span className="source-selector__icon"><Icon /></span>
            <span>
              <strong>{option.label}</strong>
              <small>{option.detail}</small>
            </span>
          </label>
        );
      })}
    </fieldset>
  );
}

function RecordingSurface({ phase, elapsed, file, onRemove }) {
  if (file) return <FileSummary file={file} onRemove={onRemove} />;

  const stateCopy = {
    idle: ["Record caller audio", "Permission is requested only when you press Start recording."],
    requesting: ["Requesting microphone", "Approve microphone access in your browser to continue."],
    recording: ["Recording in progress", "Speak naturally. Three to five seconds usually works best."],
    stopping: ["Preparing recording", "The browser is finalizing the temporary audio clip."],
  }[phase] || ["Record caller audio", "Capture a short caller-only sample."];

  return (
    <div className={`capture-surface capture-surface--${phase}`}>
      <div className="capture-surface__icon"><MicrophoneIcon /></div>
      <p className="capture-surface__eyebrow">Browser microphone</p>
      <strong>{stateCopy[0]}</strong>
      <p>{stateCopy[1]}</p>
      <div className="capture-clock" aria-label={`${Math.floor(elapsed)} seconds recorded`}>
        <span className="capture-clock__dot" />
        <time>{formatCaptureTime(elapsed)}</time>
        <span>/ {formatCaptureTime(MAX_CAPTURE_SECONDS)}</span>
      </div>
      {phase === "recording" ? (
        <div className="capture-bars" aria-hidden="true">
          {Array.from({ length: 18 }, (_, index) => <span key={index} />)}
        </div>
      ) : null}
    </div>
  );
}

function LiveSurface({ phase, stats }) {
  const copy = {
    idle: ["Stream live microphone audio", "See provisional estimates update while raw PCM chunks arrive."],
    requesting: ["Requesting microphone", "Approve access in your browser to begin the live session."],
    connecting: ["Connecting securely", "Opening the same-origin analysis WebSocket."],
    streaming: ["Listening for caller speech", "The first estimate arrives after about 1.25 seconds."],
    finalizing: ["Finalizing live result", "Microphone capture has stopped. Waiting for the settled estimate."],
    complete: ["Live session complete", "The final result is shown beside the capture controls."],
    error: ["Live session ended", "Start a new session after resolving the error below."],
  }[phase];
  const active = ["requesting", "connecting", "streaming", "finalizing"].includes(phase);

  return (
    <div className={`live-surface live-surface--${phase}`}>
      <div className="live-surface__topline">
        <div className="capture-surface__icon"><LiveIcon /></div>
        <span className={`socket-pill${active ? " socket-pill--active" : ""}`}>
          <span /> {phase === "streaming" ? "Streaming" : phase}
        </span>
      </div>
      <strong>{copy[0]}</strong>
      <p>{copy[1]}</p>
      <div
        className="mic-level"
        role="progressbar"
        aria-label="Microphone input level"
        aria-valuemin="0"
        aria-valuemax="100"
        aria-valuenow={Math.round((stats.level || 0) * 100)}
      >
        <span style={{ "--level": `${Math.max(3, (stats.level || 0) * 100)}%` }} />
      </div>
      <dl className="stream-stats">
        <div><dt>Audio</dt><dd>{formatCaptureTime(stats.elapsed || 0)}</dd></div>
        <div><dt>Chunks</dt><dd>{stats.chunks || 0}</dd></div>
        <div><dt>Sample rate</dt><dd>{stats.sampleRate ? `${(stats.sampleRate / 1000).toFixed(1)} kHz` : "—"}</dd></div>
      </dl>
      <div className="chunk-track" aria-hidden="true">
        {Array.from({ length: 16 }, (_, index) => (
          <span key={index} className={index < Math.min(16, stats.chunks || 0) ? "is-sent" : ""} />
        ))}
      </div>
      <small>Each segment represents an outgoing raw-audio batch; predictions run about once per second.</small>
    </div>
  );
}

function LiveWaitingPanel({ phase, stats }) {
  const waiting = phase === "finalizing" ? "Final inference in progress" : "Waiting for the first estimate";
  return (
    <aside className="empty-results empty-results--live" aria-label="Live stream progress">
      <p className="section-kicker">Live analysis</p>
      <h2>{waiting}</h2>
      <div className="live-orbit" aria-hidden="true"><span /><span /><span /></div>
      <p className="live-waiting-copy">
        {phase === "finalizing"
          ? "The microphone is off. The server is analyzing the complete cumulative sample."
          : "Keep speaking naturally. Provisional cards will appear here and may change as more chunks arrive."}
      </p>
      <dl className="live-waiting-stats">
        <div><dt>Captured</dt><dd>{formatCaptureTime(stats.elapsed || 0)}</dd></div>
        <div><dt>PCM chunks</dt><dd>{stats.chunks || 0}</dd></div>
      </dl>
      <div className="privacy-card">
        <LockIcon />
        <div>
          <strong>No reconnect and replay</strong>
          <p>A failed stream restarts manually, so caller audio is never retained for automatic replay.</p>
        </div>
      </div>
    </aside>
  );
}

function App() {
  const [sourceMode, setSourceMode] = useState("upload");
  const [file, setFile] = useState(null);
  const [contactId, setContactId] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [recordPhase, setRecordPhaseState] = useState("idle");
  const [recordElapsed, setRecordElapsed] = useState(0);
  const [livePhase, setLivePhase] = useState("idle");
  const [liveStats, setLiveStats] = useState({
    bytes: 0,
    chunks: 0,
    elapsed: 0,
    level: 0,
    sampleRate: 0,
  });
  const fileInputRef = useRef(null);
  const requestControllerRef = useRef(null);
  const resultsHeadingRef = useRef(null);
  const errorHeadingRef = useRef(null);
  const recorderRef = useRef(null);
  const recordPhaseRef = useRef("idle");
  const recordClockRef = useRef(null);
  const recordLimitRef = useRef(null);
  const liveSessionRef = useRef(null);

  const recordingBusy = ["requesting", "recording", "stopping"].includes(recordPhase);
  const liveBusy = ["requesting", "connecting", "streaming", "finalizing"].includes(livePhase);
  const sourceLocked = isAnalyzing || recordingBusy || liveBusy;

  useEffect(() => {
    if (result && (result.type !== "prediction" || result.is_final)) {
      resultsHeadingRef.current?.focus();
    }
  }, [result]);

  useEffect(() => {
    if (error) errorHeadingRef.current?.focus();
  }, [error]);

  useEffect(() => {
    return () => {
      const request = requestControllerRef.current;
      const recorder = recorderRef.current;
      const liveSession = liveSessionRef.current;
      requestControllerRef.current = null;
      recorderRef.current = null;
      liveSessionRef.current = null;
      request?.abort();
      recorder?.cancel();
      liveSession?.cancel();
      window.clearInterval(recordClockRef.current);
      window.clearTimeout(recordLimitRef.current);
    };
  }, []);

  function setRecordPhase(nextPhase) {
    recordPhaseRef.current = nextPhase;
    setRecordPhaseState(nextPhase);
  }

  function clearRecordClock() {
    window.clearInterval(recordClockRef.current);
    window.clearTimeout(recordLimitRef.current);
    recordClockRef.current = null;
    recordLimitRef.current = null;
  }

  function selectFile(nextFile) {
    const validationMessage = validateAudioFile(nextFile);
    if (validationMessage) {
      setError({ code: "INVALID_SELECTION", message: validationMessage });
      return;
    }
    setFile(nextFile);
    setResult(null);
    setError(null);
  }

  function clearFile() {
    const activeRequest = requestControllerRef.current;
    requestControllerRef.current = null;
    activeRequest?.abort();
    setFile(null);
    setResult(null);
    setError(null);
    setIsAnalyzing(false);
    if (sourceMode === "record" && !recordingBusy) {
      setRecordPhase("idle");
      setRecordElapsed(0);
    }
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  function startNewAnalysis() {
    recorderRef.current?.cancel();
    recorderRef.current = null;
    clearRecordClock();
    liveSessionRef.current?.cancel();
    liveSessionRef.current = null;
    clearFile();
    setContactId("");
    setRecordPhase("idle");
    setRecordElapsed(0);
    setLivePhase("idle");
    setLiveStats({ bytes: 0, chunks: 0, elapsed: 0, level: 0, sampleRate: 0 });
  }

  function changeSourceMode(nextMode) {
    if (sourceLocked || nextMode === sourceMode) return;
    recorderRef.current?.cancel();
    recorderRef.current = null;
    clearRecordClock();
    liveSessionRef.current?.cancel();
    liveSessionRef.current = null;
    setFile(null);
    setResult(null);
    setError(null);
    setIsDragging(false);
    setRecordPhase("idle");
    setRecordElapsed(0);
    setLivePhase("idle");
    setLiveStats({ bytes: 0, chunks: 0, elapsed: 0, level: 0, sampleRate: 0 });
    setSourceMode(nextMode);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  function handleDrop(event) {
    event.preventDefault();
    setIsDragging(false);
    if (!isAnalyzing && event.dataTransfer.files?.[0]) {
      selectFile(event.dataTransfer.files[0]);
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (sourceMode === "live") {
      await startLiveAnalysis();
      return;
    }
    const validationMessage = validateAudioFile(file);
    if (validationMessage) {
      setError({ code: "INVALID_SELECTION", message: validationMessage });
      return;
    }

    const controller = new AbortController();
    requestControllerRef.current = controller;
    setIsAnalyzing(true);
    setResult(null);
    setError(null);

    try {
      const response = await analyzeAudio({
        file,
        contactId,
        signal: controller.signal,
      });
      setResult(response);
    } catch (requestError) {
      if (requestError?.name !== "AbortError") {
        const normalized =
          requestError instanceof AnalysisApiError
            ? requestError
            : new AnalysisApiError("An unexpected browser error occurred.");
        setError({
          code: normalized.code,
          message: normalized.message,
          requestId: normalized.requestId,
        });
      }
    } finally {
      if (requestControllerRef.current === controller) {
        requestControllerRef.current = null;
        setIsAnalyzing(false);
      }
    }
  }

  function cancelAnalysis() {
    requestControllerRef.current?.abort();
  }

  async function startRecording() {
    if (recordPhaseRef.current !== "idle" && recordPhaseRef.current !== "ready") return;
    if (!hasRecordingSupport()) {
      setError({
        code: "RECORDING_UNSUPPORTED",
        message: "This browser does not support microphone recording. Try a current Chrome, Firefox, or Safari release.",
      });
      return;
    }
    setFile(null);
    setResult(null);
    setError(null);
    setRecordElapsed(0);
    setRecordPhase("requesting");
    const recorder = new BrowserRecorder({
      onUnexpectedStop: (captureError) => {
        if (recorderRef.current !== recorder) return;
        recorderRef.current = null;
        clearRecordClock();
        setRecordPhase("idle");
        const normalized = microphoneError(captureError);
        setError({ code: normalized.code, message: normalized.message });
      },
    });
    recorderRef.current = recorder;
    try {
      await recorder.start();
      if (recorderRef.current !== recorder) {
        recorder.cancel();
        return;
      }
      setRecordPhase("recording");
      const startedAt = performance.now();
      recordClockRef.current = window.setInterval(() => {
        setRecordElapsed(Math.min(MAX_CAPTURE_SECONDS, (performance.now() - startedAt) / 1000));
      }, 200);
      recordLimitRef.current = window.setTimeout(() => {
        void finishRecording();
      }, MAX_CAPTURE_SECONDS * 1000);
    } catch (captureError) {
      if (recorderRef.current !== recorder) return;
      recorderRef.current = null;
      setRecordPhase("idle");
      const normalized = microphoneError(captureError);
      setError({ code: normalized.code, message: normalized.message });
    }
  }

  async function finishRecording() {
    if (recordPhaseRef.current !== "recording" || !recorderRef.current) return;
    const recorder = recorderRef.current;
    clearRecordClock();
    setRecordPhase("stopping");
    try {
      const recordedFile = await recorder.stop();
      if (recorderRef.current !== recorder) return;
      recorderRef.current = null;
      setFile(recordedFile);
      setResult(null);
      setError(null);
      setRecordPhase("ready");
    } catch (captureError) {
      if (recorderRef.current !== recorder) return;
      recorderRef.current = null;
      const normalized = microphoneError(captureError);
      if (normalized.code !== "RECORDING_CANCELLED") {
        setError({ code: normalized.code, message: normalized.message });
      }
      setRecordPhase("idle");
    }
  }

  function cancelRecording() {
    clearRecordClock();
    recorderRef.current?.cancel();
    recorderRef.current = null;
    setFile(null);
    setResult(null);
    setError(null);
    setRecordElapsed(0);
    setRecordPhase("idle");
  }

  async function startLiveAnalysis() {
    if (liveSessionRef.current || liveBusy) return;
    if (contactId.trim() && !UUID_PATTERN.test(contactId.trim())) {
      setError({
        code: "INVALID_CONTACT_ID",
        message: "Contact ID must be a valid UUID, or you can leave it blank.",
      });
      return;
    }
    if (!hasLiveCaptureSupport()) {
      setError({
        code: "LIVE_UNSUPPORTED",
        message: "This browser does not support AudioWorklet microphone streaming. Try a current Chrome, Firefox, or Safari release.",
      });
      return;
    }
    setResult(null);
    setError(null);
    setLiveStats({ bytes: 0, chunks: 0, elapsed: 0, level: 0, sampleRate: 0 });

    const session = new LiveAnalysisSession({
      onError: (streamError) => {
        if (liveSessionRef.current !== session) return;
        setError({
          code: streamError.code,
          message: streamError.message,
          requestId: streamError.requestId,
        });
        setLivePhase("error");
        liveSessionRef.current = null;
      },
      onPrediction: (prediction) => {
        if (liveSessionRef.current !== session) return;
        setResult(prediction);
      },
      onState: (nextState) => {
        if (liveSessionRef.current !== session) return;
        setLivePhase(nextState === "cancelled" ? "idle" : nextState);
        if (["complete", "cancelled"].includes(nextState)) {
          liveSessionRef.current = null;
        }
      },
      onStats: (stats) => {
        if (liveSessionRef.current === session) setLiveStats(stats);
      },
    });
    liveSessionRef.current = session;
    try {
      await session.start(contactId);
    } catch {
      // The session maps microphone, connection, and service errors through onError.
    }
  }

  async function finishLiveAnalysis() {
    await liveSessionRef.current?.finish();
  }

  function cancelLiveAnalysis() {
    const session = liveSessionRef.current;
    if (!session) return;
    liveSessionRef.current = null;
    setResult(null);
    setError(null);
    setLivePhase("idle");
    setLiveStats({ bytes: 0, chunks: 0, elapsed: 0, level: 0, sampleRate: 0 });
    void session.cancel();
  }

  function renderActions() {
    if (isAnalyzing) {
      return (
        <>
          <button className="primary-button" type="submit" disabled>
            <span className="spinner" /> Analyzing securely…
          </button>
          <button className="text-button" type="button" onClick={cancelAnalysis}>Cancel</button>
        </>
      );
    }
    if (sourceMode === "upload") {
      return <button className="primary-button" type="submit" disabled={!file}>Analyze audio</button>;
    }
    if (sourceMode === "record") {
      if (recordPhase === "recording") {
        return (
          <>
            <button className="primary-button primary-button--stop" type="button" onClick={finishRecording}>Stop recording</button>
            <button className="text-button" type="button" onClick={cancelRecording}>Discard</button>
          </>
        );
      }
      if (recordPhase === "requesting" || recordPhase === "stopping") {
        return (
          <>
            <button className="primary-button" type="button" disabled>
              <span className="spinner" /> {recordPhase === "requesting" ? "Waiting for permission…" : "Preparing audio…"}
            </button>
            <button className="text-button" type="button" onClick={cancelRecording}>
              {recordPhase === "requesting" ? "Cancel request" : "Discard recording"}
            </button>
          </>
        );
      }
      if (file) {
        return (
          <>
            <button className="primary-button" type="submit">Analyze recording</button>
            <button className="text-button" type="button" onClick={cancelRecording}>Record again</button>
          </>
        );
      }
      return <button className="primary-button" type="button" onClick={startRecording}>Start recording</button>;
    }
    if (livePhase === "streaming") {
      return (
        <>
          <button className="primary-button primary-button--stop" type="button" onClick={finishLiveAnalysis}>Stop &amp; finalize</button>
          <button className="text-button" type="button" onClick={cancelLiveAnalysis}>Cancel stream</button>
        </>
      );
    }
    if (["requesting", "connecting", "finalizing"].includes(livePhase)) {
      return (
        <>
          <button className="primary-button" type="button" disabled>
            <span className="spinner" /> {livePhase === "finalizing" ? "Finalizing result…" : "Starting live stream…"}
          </button>
          <button className="text-button" type="button" onClick={cancelLiveAnalysis}>
            {livePhase === "finalizing" ? "Cancel finalization" : "Cancel request"}
          </button>
        </>
      );
    }
    if (livePhase === "complete") return null;
    return (
      <button className="primary-button" type="button" onClick={startLiveAnalysis}>
        {livePhase === "error" ? "Retry live analysis" : "Start live analysis"}
      </button>
    );
  }

  const panelTitles = {
    live: "Stream caller audio",
    record: "Record caller audio",
    upload: "Choose caller audio",
  };
  const liveHasActivity = ["requesting", "connecting", "streaming", "finalizing"].includes(livePhase);
  const captureStatus =
    sourceMode === "record"
      ? {
          idle: "Recorder ready.",
          requesting: "Waiting for microphone permission.",
          recording: "Microphone recording started.",
          stopping: "Microphone is off. Preparing the recording.",
          ready: "Recording ready to analyze.",
        }[recordPhase]
      : sourceMode === "live"
        ? {
            idle: "Live analyzer ready.",
            requesting: "Waiting for microphone permission.",
            connecting: "Connecting the live audio stream.",
            streaming: "Live microphone streaming started.",
            finalizing: "Microphone is off. Waiting for the final result.",
            complete: "Final live result received.",
            error: "Live analysis ended with an error.",
          }[livePhase]
        : "";

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="/" aria-label="Diaflow Voice Analyzer home">
          <span className="brand__mark">
            <WaveIcon />
          </span>
          <span>
            <strong>Diaflow</strong>
            <small>Voice intelligence</small>
          </span>
        </a>
        <ServiceStatus />
      </header>

      <main>
        <section className="hero">
          <div>
            <p className="hero__eyebrow">Logistics voice operations</p>
            <h1>Analyze a caller voice sample in seconds.</h1>
          </div>
          <p className="hero__copy">
            Upload, record, or stream caller-only audio to estimate perceived voice
            presentation and an adult age bracket—with quality flags and explicit abstention.
          </p>
        </section>

        <div className="workspace-grid">
          <section className="upload-panel" aria-labelledby="upload-title">
            <div className="panel-heading">
              <div>
                <p className="section-kicker">New analysis</p>
                <h2 id="upload-title">{panelTitles[sourceMode]}</h2>
              </div>
              <span className="step-pill">Step 1 of 1</span>
            </div>

            <form onSubmit={handleSubmit}>
              <SourceSelector value={sourceMode} disabled={sourceLocked} onChange={changeSourceMode} />

              {sourceMode === "upload" && !file ? (
                <div
                  className={`dropzone${isDragging ? " dropzone--active" : ""}`}
                  onDragEnter={(event) => {
                    event.preventDefault();
                    if (!isAnalyzing) setIsDragging(true);
                  }}
                  onDragOver={(event) => event.preventDefault()}
                  onDragLeave={(event) => {
                    if (!event.currentTarget.contains(event.relatedTarget)) {
                      setIsDragging(false);
                    }
                  }}
                  onDrop={handleDrop}
                >
                  <div className="dropzone__icon">
                    <UploadIcon />
                  </div>
                  <strong>Drop an audio recording here</strong>
                  <p>M4A works directly—no conversion required.</p>
                  <input
                    ref={fileInputRef}
                    id="audio-file"
                    type="file"
                    accept="audio/*,.m4a,.mp4,.wav,.mp3,.flac,.ogg,.opus,.webm"
                    onChange={(event) => {
                      if (event.target.files?.[0]) selectFile(event.target.files[0]);
                    }}
                  />
                  <label className="file-picker" htmlFor="audio-file">
                    Browse files
                  </label>
                  <span className="dropzone__limits">Maximum 12 MB · Best with 3–5 seconds</span>
                </div>
              ) : sourceMode === "upload" ? (
                <FileSummary file={file} onRemove={clearFile} />
              ) : sourceMode === "record" ? (
                <RecordingSurface
                  phase={recordPhase}
                  elapsed={recordElapsed}
                  file={file}
                  onRemove={clearFile}
                />
              ) : (
                <LiveSurface phase={livePhase} stats={liveStats} />
              )}

              <div className="field-group">
                <label htmlFor="contact-id">
                  Contact ID <span>Optional</span>
                </label>
                <input
                  id="contact-id"
                  type="text"
                  value={contactId}
                  onChange={(event) => setContactId(event.target.value)}
                  placeholder="Generated automatically when blank"
                  autoComplete="off"
                  disabled={sourceLocked}
                />
                <small>Use an opaque UUID only—never a name or phone number.</small>
              </div>

              {error ? (
                <div className="error-alert" role="alert">
                  <div>
                    <strong ref={errorHeadingRef} tabIndex="-1">
                      Analysis could not be completed
                    </strong>
                    <p>{error.message}</p>
                    {error.requestId ? <small>Request ID: {error.requestId}</small> : null}
                  </div>
                </div>
              ) : null}

              <div className="form-actions">
                {renderActions()}
              </div>
              {captureStatus ? <p className="sr-only" role="status">{captureStatus}</p> : null}
            </form>
          </section>

          <div className="result-region">
            {result ? (
              <ResultsPanel
                result={result}
                headingRef={resultsHeadingRef}
                onReset={startNewAnalysis}
                streamStats={liveStats}
              />
            ) : sourceMode === "live" && liveHasActivity ? (
              <LiveWaitingPanel phase={livePhase} stats={liveStats} />
            ) : (
              <EmptyResults />
            )}
          </div>
        </div>
        {result?.type === "prediction" && !result.is_final ? (
          <p className="sr-only" role="status">Live estimate updated, sequence {result.sequence}.</p>
        ) : null}
      </main>

      <footer>
        <span>Diaflow Voice Analyzer</span>
        <p>Audio remains request-scoped and is not stored by the application.</p>
      </footer>
    </div>
  );
}

export default App;
