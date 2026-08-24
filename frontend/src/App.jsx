import { useEffect, useRef, useState } from "react";

import {
  AnalysisApiError,
  analyzeAudio,
  formatBytes,
  validateAudioFile,
} from "./api/analyze.js";

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

function ResultsPanel({ result, headingRef, onReset }) {
  const [copied, setCopied] = useState(false);
  const quality = QUALITY_COPY[result.audio_quality] || QUALITY_COPY.degraded;

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
          <p className="section-kicker">Analysis complete</p>
          <h2 id="results-title" ref={headingRef} tabIndex="-1">
            Contact attributes
          </h2>
        </div>
        <button className="secondary-button" type="button" onClick={onReset}>
          Analyze another
        </button>
      </div>

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

function App() {
  const [file, setFile] = useState(null);
  const [contactId, setContactId] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const fileInputRef = useRef(null);
  const requestControllerRef = useRef(null);
  const resultsHeadingRef = useRef(null);
  const errorHeadingRef = useRef(null);

  useEffect(() => {
    if (result) resultsHeadingRef.current?.focus();
  }, [result]);

  useEffect(() => {
    if (error) errorHeadingRef.current?.focus();
  }, [error]);

  useEffect(() => {
    return () => requestControllerRef.current?.abort();
  }, []);

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
    requestControllerRef.current?.abort();
    setFile(null);
    setResult(null);
    setError(null);
    setIsAnalyzing(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  function startNewAnalysis() {
    clearFile();
    setContactId("");
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
            Upload caller-only audio to estimate perceived voice presentation and an adult
            age bracket—with quality flags and explicit abstention when confidence is low.
          </p>
        </section>

        <div className="workspace-grid">
          <section className="upload-panel" aria-labelledby="upload-title">
            <div className="panel-heading">
              <div>
                <p className="section-kicker">New analysis</p>
                <h2 id="upload-title">Choose caller audio</h2>
              </div>
              <span className="step-pill">Step 1 of 1</span>
            </div>

            <form onSubmit={handleSubmit}>
              {!file ? (
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
              ) : (
                <FileSummary file={file} onRemove={clearFile} />
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
                  disabled={isAnalyzing}
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
                <button
                  className="primary-button"
                  type="submit"
                  disabled={!file || isAnalyzing}
                >
                  {isAnalyzing ? (
                    <>
                      <span className="spinner" /> Analyzing securely…
                    </>
                  ) : (
                    "Analyze audio"
                  )}
                </button>
                {isAnalyzing ? (
                  <button className="text-button" type="button" onClick={cancelAnalysis}>
                    Cancel
                  </button>
                ) : null}
              </div>
            </form>
          </section>

          <div className="result-region" aria-live="polite">
            {result ? (
              <ResultsPanel
                result={result}
                headingRef={resultsHeadingRef}
                onReset={startNewAnalysis}
              />
            ) : (
              <EmptyResults />
            )}
          </div>
        </div>
      </main>

      <footer>
        <span>Diaflow Voice Analyzer</span>
        <p>Audio remains request-scoped and is not stored by the application.</p>
      </footer>
    </div>
  );
}

export default App;
