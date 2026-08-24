import { useEffect, useRef, useState } from "react";

import {
  AnalysisApiError,
  analyzeAudio,
  formatBytes,
  languageName,
  validateAudioFile,
} from "./api/analyze.js";
import {
  PERSISTENCE_MODES,
  PersistenceApiError,
  deleteStoredAnalysis,
  getPersistenceCapabilities,
  getStoredAnalysis,
  listStoredAnalyses,
  persistenceModeIsAvailable,
} from "./api/persistence.js";
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

const PERSISTENCE_OPTIONS = [
  {
    value: PERSISTENCE_MODES.NONE,
    label: "Do not store",
    detail: "Request-scoped audio and result",
  },
  {
    value: PERSISTENCE_MODES.RESULT,
    label: "Store result",
    detail: "Attributes and operational metadata only",
  },
  {
    value: PERSISTENCE_MODES.RESULT_AND_AUDIO,
    label: "Store result + audio",
    detail: "Requires caller consent or an approved retention basis",
  },
];

const PERSISTENCE_LABELS = {
  [PERSISTENCE_MODES.NONE]: "Not stored",
  [PERSISTENCE_MODES.RESULT]: "Result only",
  [PERSISTENCE_MODES.RESULT_AND_AUDIO]: "Result + audio",
};

function storedAnalysisId(analysis) {
  return (
    analysis?.analysis_id ||
    analysis?.session_id ||
    analysis?.persistence?.session_id ||
    analysis?.id ||
    ""
  );
}

function storedAnalysisResult(analysis) {
  return analysis?.result && typeof analysis.result === "object" ? analysis.result : analysis;
}

function formatStoredDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

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

function PersistenceSummary({ requestedMode, result, streamStats }) {
  const raw = result.persistence;
  const persistence = raw && typeof raw === "object" ? raw : {};
  const candidateMode = persistence.mode || requestedMode || PERSISTENCE_MODES.NONE;
  const mode = typeof candidateMode === "string" ? candidateMode : PERSISTENCE_MODES.NONE;
  const candidateAnalysisId =
    result.analysis_id || persistence.analysis_id || persistence.session_id || "";
  const analysisId = typeof candidateAnalysisId === "string" ? candidateAnalysisId : "";
  const candidateStatus =
    persistence.status ||
    (typeof raw === "string" ? raw : "") ||
    (analysisId ? "completed" : mode === PERSISTENCE_MODES.NONE ? "not stored" : "requested");
  const status = typeof candidateStatus === "string" ? candidateStatus : "requested";
  const chunksReceived = persistence.chunks_received;
  const chunksStored = persistence.chunks_stored;
  const segmentCount = persistence.segments_stored ?? persistence.segment_count;
  const audioBytes = persistence.bytes_stored ?? persistence.audio_bytes;
  const legacyExpiresAt = persistence.expires_at;
  const audioExpiresAt =
    persistence.audio_expires_at ??
    (mode === PERSISTENCE_MODES.RESULT_AND_AUDIO ? legacyExpiresAt : null);
  const resultExpiresAt =
    persistence.result_expires_at ??
    (mode === PERSISTENCE_MODES.RESULT ? legacyExpiresAt : null);
  const shouldShow = mode !== PERSISTENCE_MODES.NONE || Boolean(raw) || Boolean(analysisId);
  if (!shouldShow) return null;

  const inProgress = ["pending", "requested", "storing", "streaming"].includes(status);
  const failed = status === "failed";
  const partial = status === "partial";
  const progressValue =
    Number.isFinite(chunksReceived) && chunksReceived > 0 && Number.isFinite(chunksStored)
      ? Math.min(100, Math.round((chunksStored / chunksReceived) * 100))
      : failed
        ? 0
        : inProgress
          ? undefined
          : 100;
  const progressLabel = failed
    ? "Storage failed"
    : partial
      ? "Storage partially completed"
    : inProgress
      ? "Storage in progress"
      : "Storage confirmed";

  return (
    <section className={`persistence-summary persistence-summary--${failed ? "failed" : status}`}>
      <div className="persistence-summary__heading">
        <div>
          <span>Storage progress</span>
          <strong>{progressLabel}</strong>
        </div>
        <span className="storage-status">{status.replaceAll("_", " ")}</span>
      </div>
      <div
        className={`storage-progress${inProgress ? " storage-progress--active" : ""}`}
        role="progressbar"
        aria-label="Server-side storage progress"
        aria-valuemin="0"
        aria-valuemax="100"
        aria-valuenow={progressValue}
        aria-valuetext={progressLabel}
      >
        <span />
      </div>
      <dl>
        <div><dt>Retention</dt><dd>{PERSISTENCE_LABELS[mode] || mode}</dd></div>
        {analysisId ? <div><dt>Analysis ID</dt><dd title={analysisId}>{analysisId}</dd></div> : null}
        {Number.isFinite(chunksReceived) ? <div><dt>Chunks received</dt><dd>{chunksReceived}</dd></div> : null}
        {Number.isFinite(chunksStored) ? <div><dt>Chunks stored</dt><dd>{chunksStored}</dd></div> : null}
        {Number.isFinite(segmentCount) ? <div><dt>Segments stored</dt><dd>{segmentCount}</dd></div> : null}
        {Number.isFinite(audioBytes) ? <div><dt>Audio stored</dt><dd>{formatBytes(audioBytes)}</dd></div> : null}
        {audioExpiresAt ? <div><dt>Audio expires</dt><dd>{formatStoredDate(audioExpiresAt)}</dd></div> : null}
        {resultExpiresAt ? <div><dt>Result expires</dt><dd>{formatStoredDate(resultExpiresAt)}</dd></div> : null}
        {mode === PERSISTENCE_MODES.RESULT_AND_AUDIO && inProgress && !Number.isFinite(segmentCount) ? (
          <div><dt>Chunks sent</dt><dd>{streamStats?.chunks || 0}</dd></div>
        ) : null}
      </dl>
      {!raw && mode !== PERSISTENCE_MODES.NONE ? (
        <p>The request was sent with this retention choice; this server did not return storage metadata.</p>
      ) : null}
    </section>
  );
}

function ResultsPanel({ result, headingRef, onReset, persistenceMode, storage, streamStats }) {
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
        {result.language ? (
          <PredictionCard
            label="Spoken language"
            prediction={{
              ...result.language,
              prediction: languageName(result.language.prediction),
            }}
          />
        ) : null}
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

      <PersistenceSummary
        requestedMode={persistenceMode}
        result={
          storage && result.type === "prediction"
            ? {
                ...result,
                analysis_id: result.analysis_id || storage.analysis_id,
                persistence: result.persistence || storage.persistence,
              }
            : result
        }
        streamStats={streamStats}
      />

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
          <strong>Ephemeral by default</strong>
          <p>Nothing is stored unless you explicitly choose retention for this analysis.</p>
        </div>
      </div>
    </aside>
  );
}

function StoredAnalysisDetail({ analysis, loading }) {
  const result = storedAnalysisResult(analysis);
  const persistence = analysis?.persistence || {};
  const id = storedAnalysisId(analysis);
  const gender = result?.gender;
  const age = result?.age_bracket;
  const mode = persistence.mode || analysis?.mode || PERSISTENCE_MODES.RESULT;
  const status = persistence.status || analysis?.status || "completed";
  const segments = Array.isArray(analysis?.segments) ? analysis.segments : [];
  const audioExpiresAt = persistence.audio_expires_at;
  const resultExpiresAt = persistence.result_expires_at;
  const genericExpiresAt = persistence.expires_at || analysis?.expires_at;
  const hasSpecificExpiry = Boolean(audioExpiresAt || resultExpiresAt);

  function predictionCopy(prediction) {
    if (!prediction || typeof prediction.prediction !== "string") return "—";
    if (!Number.isFinite(prediction.confidence)) return prediction.prediction;
    return `${prediction.prediction} · ${Math.round(prediction.confidence * 100)}%`;
  }

  return (
    <aside className="history-detail" aria-label="Stored analysis detail">
      <div className="history-detail__heading">
        <div>
          <span>Selected analysis</span>
          <strong title={id}>{id || "Identifier unavailable"}</strong>
        </div>
        <span className="storage-status">{String(status).replaceAll("_", " ")}</span>
      </div>
      <dl>
        <div><dt>Contact</dt><dd title={result?.contact_id}>{result?.contact_id || analysis?.contact_id || "—"}</dd></div>
        <div><dt>Created</dt><dd>{formatStoredDate(analysis?.created_at || persistence.created_at)}</dd></div>
        <div><dt>Voice presentation</dt><dd>{predictionCopy(gender)}</dd></div>
        <div><dt>Age bracket</dt><dd>{predictionCopy(age)}</dd></div>
        {result?.language ? (
          <div>
            <dt>Spoken language</dt>
            <dd>
              {predictionCopy({
                ...result.language,
                prediction: languageName(result.language.prediction),
              })}
            </dd>
          </div>
        ) : null}
        <div><dt>Audio quality</dt><dd>{result?.audio_quality || "—"}</dd></div>
        <div><dt>Retention</dt><dd>{PERSISTENCE_LABELS[mode] || mode}</dd></div>
        {Number.isFinite(persistence.segments_stored ?? persistence.segment_count ?? analysis?.segment_count) ? (
          <div><dt>Segments</dt><dd>{persistence.segments_stored ?? persistence.segment_count ?? analysis.segment_count}</dd></div>
        ) : null}
        {Number.isFinite(persistence.bytes_stored ?? persistence.audio_bytes ?? analysis?.audio_bytes) ? (
          <div><dt>Audio stored</dt><dd>{formatBytes(persistence.bytes_stored ?? persistence.audio_bytes ?? analysis.audio_bytes)}</dd></div>
        ) : null}
        {audioExpiresAt ? (
          <div><dt>Audio expires</dt><dd>{formatStoredDate(audioExpiresAt)}</dd></div>
        ) : null}
        {resultExpiresAt ? (
          <div><dt>Result expires</dt><dd>{formatStoredDate(resultExpiresAt)}</dd></div>
        ) : null}
        {!hasSpecificExpiry && genericExpiresAt ? (
          <div>
            <dt>{mode === PERSISTENCE_MODES.RESULT_AND_AUDIO ? "Audio + result expire" : "Result expires"}</dt>
            <dd>{formatStoredDate(genericExpiresAt)}</dd>
          </div>
        ) : null}
      </dl>
      {loading ? <p className="segment-loading" role="status">Loading stored segment detail…</p> : null}
      {segments.length ? (
        <div className="segment-detail">
          <strong>Stored audio segments</strong>
          <ol>
            {segments.map((segment, index) => (
              <li key={`${segment.object_key || "segment"}-${segment.sequence ?? index}`}>
                <div>
                  <strong>Segment {Number.isFinite(segment.sequence) ? segment.sequence : index}</strong>
                  <span>{Number.isFinite(segment.byte_size) ? formatBytes(segment.byte_size) : "—"}</span>
                </div>
                <code title={segment.object_key}>{segment.object_key || "Object key unavailable"}</code>
                {Number.isFinite(segment.byte_start) && Number.isFinite(segment.byte_end) ? (
                  <p>Stored byte range {segment.byte_start}–{segment.byte_end}</p>
                ) : null}
                {Array.isArray(segment.logical_chunks) && segment.logical_chunks.length ? (
                  <ul>
                    {segment.logical_chunks.map((chunk) => (
                      <li key={`${chunk.chunk_index}-${chunk.segment_byte_start}`}>
                        Chunk {chunk.chunk_index}: source {chunk.source_byte_start}–{chunk.source_byte_end}; segment {chunk.segment_byte_start}–{chunk.segment_byte_end}
                      </li>
                    ))}
                  </ul>
                ) : null}
              </li>
            ))}
          </ol>
        </div>
      ) : null}
    </aside>
  );
}

function StoredAnalysisHistory({ refreshToken }) {
  const [open, setOpen] = useState(false);
  const [reloadToken, setReloadToken] = useState(0);
  const [phase, setPhase] = useState("idle");
  const [analyses, setAnalyses] = useState([]);
  const [selectedId, setSelectedId] = useState("");
  const [selectedDetail, setSelectedDetail] = useState(null);
  const [detailPhase, setDetailPhase] = useState("idle");
  const [confirmDeleteId, setConfirmDeleteId] = useState("");
  const [deletingId, setDeletingId] = useState("");
  const [historyError, setHistoryError] = useState("");
  const [notice, setNotice] = useState("");
  const detailControllerRef = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    const controller = new AbortController();

    listStoredAnalyses({ signal: controller.signal })
      .then((items) => {
        setAnalyses(items);
        setSelectedId((current) =>
          current && items.some((item) => storedAnalysisId(item) === current) ? current : "",
        );
        setPhase("ready");
      })
      .catch((requestError) => {
        if (requestError?.name === "AbortError") return;
        const normalized =
          requestError instanceof PersistenceApiError
            ? requestError
            : new PersistenceApiError("Stored-analysis history could not be loaded.");
        setHistoryError(normalized.message);
        setPhase("error");
      });
    return () => controller.abort();
  }, [open, refreshToken, reloadToken]);

  useEffect(() => () => detailControllerRef.current?.abort(), []);

  const selectedSummary = analyses.find((analysis) => storedAnalysisId(analysis) === selectedId);
  const selected =
    selectedDetail && storedAnalysisId(selectedDetail) === selectedId
      ? selectedDetail
      : selectedSummary;

  function toggleHistory() {
    if (!open) {
      setPhase("loading");
      setHistoryError("");
      setNotice("");
    }
    if (open) detailControllerRef.current?.abort();
    setOpen((value) => !value);
  }

  async function selectAnalysis(analysisId) {
    detailControllerRef.current?.abort();
    setSelectedId(analysisId);
    setSelectedDetail(null);
    if (!analysisId) return;

    const controller = new AbortController();
    detailControllerRef.current = controller;
    setDetailPhase("loading");
    setHistoryError("");
    try {
      const detail = await getStoredAnalysis(analysisId, { signal: controller.signal });
      if (detailControllerRef.current !== controller) return;
      setSelectedDetail(detail);
      setDetailPhase("ready");
    } catch (requestError) {
      if (requestError?.name === "AbortError") return;
      const normalized =
        requestError instanceof PersistenceApiError
          ? requestError
          : new PersistenceApiError("Stored-analysis detail could not be loaded.");
      setHistoryError(normalized.message);
      setDetailPhase("error");
    } finally {
      if (detailControllerRef.current === controller) detailControllerRef.current = null;
    }
  }

  function refreshHistory() {
    setPhase("loading");
    setHistoryError("");
    setNotice("");
    setReloadToken((value) => value + 1);
  }

  async function removeAnalysis(analysisId) {
    if (confirmDeleteId !== analysisId) {
      setConfirmDeleteId(analysisId);
      setNotice("Press Confirm delete to permanently remove this stored analysis and its retained audio.");
      return;
    }
    setDeletingId(analysisId);
    setHistoryError("");
    setNotice("");
    try {
      await deleteStoredAnalysis(analysisId);
      setAnalyses((current) => current.filter((item) => storedAnalysisId(item) !== analysisId));
      if (selectedId === analysisId) {
        setSelectedId("");
        setSelectedDetail(null);
      }
      setConfirmDeleteId("");
      setNotice("Stored analysis deleted.");
    } catch (requestError) {
      const normalized =
        requestError instanceof PersistenceApiError
          ? requestError
          : new PersistenceApiError("The stored analysis could not be deleted.");
      setHistoryError(normalized.message);
    } finally {
      setDeletingId("");
    }
  }

  return (
    <section className="history-panel" aria-labelledby="history-title">
      <div className="history-panel__heading">
        <div>
          <p className="section-kicker">Retention controls</p>
          <h2 id="history-title">Stored analyses</h2>
          <p>Review server-side records and explicitly remove results or retained audio.</p>
        </div>
        <button className="secondary-button" type="button" onClick={toggleHistory}>
          {open ? "Hide history" : "View history"}
        </button>
      </div>

      {open ? (
        <div className="history-panel__body">
          <div className="history-toolbar">
            <span>{phase === "ready" ? `${analyses.length} stored` : "Server-side history"}</span>
            <button
              className="text-button"
              type="button"
              disabled={phase === "loading"}
              onClick={refreshHistory}
            >
              {phase === "loading" ? "Refreshing…" : "Refresh"}
            </button>
          </div>
          {historyError ? <p className="history-message history-message--error" role="alert">{historyError}</p> : null}
          {notice ? <p className="history-message" role="status">{notice}</p> : null}
          {phase === "loading" && !analyses.length ? (
            <div className="history-loading" role="status"><span className="spinner" /> Loading stored analyses…</div>
          ) : null}
          {phase === "ready" && !analyses.length ? (
            <p className="history-empty">No stored analyses were returned. New requests still default to no retention.</p>
          ) : null}
          {analyses.length ? (
            <div className="history-layout">
              <ul className="history-list">
                {analyses.map((analysis, index) => {
                  const id = storedAnalysisId(analysis);
                  const itemResult = storedAnalysisResult(analysis);
                  const createdAt = analysis?.created_at || analysis?.persistence?.created_at;
                  return (
                    <li key={id || `${itemResult?.contact_id || "analysis"}-${index}`} className={selectedId === id ? "is-selected" : ""}>
                      <button
                        className="history-list__select"
                        type="button"
                        onClick={() => void selectAnalysis(id)}
                        disabled={!id}
                      >
                        <strong>{itemResult?.contact_id || analysis?.contact_id || "Unknown contact"}</strong>
                        <span>{formatStoredDate(createdAt)}</span>
                        <small>{id || "Missing analysis ID"}</small>
                      </button>
                      <button
                        className={`history-list__delete${confirmDeleteId === id ? " is-confirming" : ""}`}
                        type="button"
                        disabled={!id || deletingId === id}
                        onClick={() => void removeAnalysis(id)}
                      >
                        {deletingId === id
                          ? "Deleting…"
                          : confirmDeleteId === id
                            ? "Confirm delete"
                            : "Delete"}
                      </button>
                    </li>
                  );
                })}
              </ul>
              {selected ? <StoredAnalysisDetail analysis={selected} loading={detailPhase === "loading"} /> : (
                <div className="history-detail history-detail--empty">Select an analysis to inspect its stored result and retention metadata.</div>
              )}
            </div>
          ) : null}
        </div>
      ) : null}
    </section>
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

function PersistenceControls({
  acknowledged,
  capabilitiesPhase,
  consentReference,
  disabled,
  maximumMode,
  mode,
  onAcknowledgedChange,
  onConsentReferenceChange,
  onModeChange,
}) {
  const storesAudio = mode === PERSISTENCE_MODES.RESULT_AND_AUDIO;
  const ready = !storesAudio || (acknowledged && consentReference.trim());
  const availableOptions = PERSISTENCE_OPTIONS.filter((option) =>
    persistenceModeIsAvailable(option.value, maximumMode),
  );
  const availabilityCopy =
    capabilitiesPhase === "loading"
      ? "Checking server storage"
      : capabilitiesPhase === "error"
        ? "Storage unavailable · default none"
        : maximumMode === PERSISTENCE_MODES.NONE
          ? "Server retention disabled"
          : "Default: none";

  return (
    <fieldset className="persistence-controls" disabled={disabled}>
      <div className="persistence-controls__heading">
        <legend>Storage mode</legend>
        <span role="status">{availabilityCopy}</span>
      </div>
      <div className="persistence-options">
        {availableOptions.map((option) => (
          <label key={option.value} className={mode === option.value ? "is-selected" : ""}>
            <input
              type="radio"
              name="persistence-mode"
              value={option.value}
              checked={mode === option.value}
              onChange={() => onModeChange(option.value)}
            />
            <span className="persistence-options__control" />
            <span>
              <strong>{option.label}</strong>
              <small>{option.detail}</small>
            </span>
          </label>
        ))}
      </div>

      {storesAudio ? (
        <div className="consent-panel">
          <label className="consent-check">
            <input
              type="checkbox"
              checked={acknowledged}
              onChange={(event) => onAcknowledgedChange(event.target.checked)}
            />
            <span>
              I confirm caller consent or another approved basis for retaining this audio.
            </span>
          </label>
          <label htmlFor="consent-reference">
            Consent reference <span>Required</span>
          </label>
          <input
            id="consent-reference"
            type="text"
            value={consentReference}
            onChange={(event) => onConsentReferenceChange(event.target.value)}
            placeholder="Approval, ticket, or policy reference"
            autoComplete="off"
            maxLength="160"
          />
          <small>Use an opaque internal reference—never a name, phone number, or transcript.</small>
          {!ready ? <p role="status">A confirmation and reference are required before audio can be sent for storage.</p> : null}
        </div>
      ) : null}
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

function LiveWaitingPanel({ persistenceMode, phase, stats, storage }) {
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
      {storage ? (
        <PersistenceSummary
          requestedMode={persistenceMode}
          result={storage}
          streamStats={stats}
        />
      ) : (
        <div className="privacy-card">
          <LockIcon />
          <div>
            <strong>No automatic replay</strong>
            <p>A failed stream restarts manually. Server retention follows only the explicit storage choice for this session.</p>
          </div>
        </div>
      )}
    </aside>
  );
}

function App() {
  const [sourceMode, setSourceMode] = useState("upload");
  const [file, setFile] = useState(null);
  const [contactId, setContactId] = useState("");
  const [persistenceMode, setPersistenceMode] = useState(PERSISTENCE_MODES.NONE);
  const [persistenceCapabilities, setPersistenceCapabilities] = useState({
    enabled: false,
    maximum_mode: PERSISTENCE_MODES.NONE,
    default_mode: PERSISTENCE_MODES.NONE,
  });
  const [capabilitiesPhase, setCapabilitiesPhase] = useState("loading");
  const [consentAcknowledged, setConsentAcknowledged] = useState(false);
  const [consentReference, setConsentReference] = useState("");
  const [activePersistenceMode, setActivePersistenceMode] = useState(PERSISTENCE_MODES.NONE);
  const [historyRevision, setHistoryRevision] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [recordPhase, setRecordPhaseState] = useState("idle");
  const [recordElapsed, setRecordElapsed] = useState(0);
  const [livePhase, setLivePhase] = useState("idle");
  const [liveStorage, setLiveStorage] = useState(null);
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
  const maximumPersistenceMode = persistenceCapabilities.maximum_mode;
  const persistenceReady =
    persistenceMode !== PERSISTENCE_MODES.RESULT_AND_AUDIO ||
    (consentAcknowledged && Boolean(consentReference.trim()));

  useEffect(() => {
    if (result && (result.type !== "prediction" || result.is_final)) {
      resultsHeadingRef.current?.focus();
    }
  }, [result]);

  useEffect(() => {
    if (error) errorHeadingRef.current?.focus();
  }, [error]);

  useEffect(() => {
    const controller = new AbortController();
    getPersistenceCapabilities({ signal: controller.signal })
      .then((capabilities) => {
        if (controller.signal.aborted) return;
        setPersistenceCapabilities(capabilities);
        setCapabilitiesPhase("ready");
      })
      .catch((capabilitiesError) => {
        if (capabilitiesError?.name === "AbortError" || controller.signal.aborted) return;
        setPersistenceCapabilities({
          enabled: false,
          maximum_mode: PERSISTENCE_MODES.NONE,
          default_mode: PERSISTENCE_MODES.NONE,
        });
        setCapabilitiesPhase("error");
      });
    return () => controller.abort();
  }, []);

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
    setPersistenceMode(PERSISTENCE_MODES.NONE);
    setConsentAcknowledged(false);
    setConsentReference("");
    setActivePersistenceMode(PERSISTENCE_MODES.NONE);
    setRecordPhase("idle");
    setRecordElapsed(0);
    setLivePhase("idle");
    setLiveStorage(null);
    setLiveStats({ bytes: 0, chunks: 0, elapsed: 0, level: 0, sampleRate: 0 });
  }

  function changePersistenceMode(nextMode) {
    if (!persistenceModeIsAvailable(nextMode, maximumPersistenceMode)) return;
    setPersistenceMode(nextMode);
    setError(null);
    if (nextMode !== PERSISTENCE_MODES.RESULT_AND_AUDIO) {
      setConsentAcknowledged(false);
      setConsentReference("");
    }
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
    setLiveStorage(null);
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
    if (!persistenceReady) {
      setError({
        code: "PERSISTENCE_CONSENT_REQUIRED",
        message: "Confirm the approved audio-retention basis and provide its opaque reference.",
      });
      return;
    }
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
    setActivePersistenceMode(persistenceMode);
    setResult(null);
    setError(null);
    setLiveStorage(null);

    try {
      const response = await analyzeAudio({
        file,
        contactId,
        persistenceMode,
        consentReference,
        signal: controller.signal,
      });
      setResult(response);
      if (persistenceMode !== PERSISTENCE_MODES.NONE) {
        setHistoryRevision((value) => value + 1);
      }
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
    setLiveStorage(null);
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
    if (!persistenceReady) {
      setError({
        code: "PERSISTENCE_CONSENT_REQUIRED",
        message: "Confirm the approved audio-retention basis and provide its opaque reference.",
      });
      return;
    }
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
    setLiveStorage(null);
    setActivePersistenceMode(persistenceMode);
    setLiveStats({ bytes: 0, chunks: 0, elapsed: 0, level: 0, sampleRate: 0 });

    const session = new LiveAnalysisSession({
      consentReference,
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
        if (prediction.is_final && persistenceMode !== PERSISTENCE_MODES.NONE) {
          setHistoryRevision((value) => value + 1);
        }
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
      onStorage: (storage) => {
        if (liveSessionRef.current === session) setLiveStorage(storage);
      },
      persistenceMode,
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
    setLiveStorage(null);
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
      return <button className="primary-button" type="submit" disabled={!file || !persistenceReady}>Analyze audio</button>;
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
            <button className="primary-button" type="submit" disabled={!persistenceReady}>Analyze recording</button>
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
      <button className="primary-button" type="button" onClick={startLiveAnalysis} disabled={!persistenceReady}>
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

              <PersistenceControls
                acknowledged={consentAcknowledged}
                capabilitiesPhase={capabilitiesPhase}
                consentReference={consentReference}
                disabled={sourceLocked}
                maximumMode={maximumPersistenceMode}
                mode={persistenceMode}
                onAcknowledgedChange={setConsentAcknowledged}
                onConsentReferenceChange={setConsentReference}
                onModeChange={changePersistenceMode}
              />

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
                persistenceMode={activePersistenceMode}
                storage={liveStorage}
                streamStats={liveStats}
              />
            ) : sourceMode === "live" && liveHasActivity ? (
              <LiveWaitingPanel
                persistenceMode={activePersistenceMode}
                phase={livePhase}
                stats={liveStats}
                storage={liveStorage}
              />
            ) : (
              <EmptyResults />
            )}
          </div>
        </div>
        <StoredAnalysisHistory refreshToken={historyRevision} />
        {result?.type === "prediction" && !result.is_final ? (
          <p className="sr-only" role="status">Live estimate updated, sequence {result.sequence}.</p>
        ) : null}
      </main>

      <footer>
        <span>Diaflow Voice Analyzer</span>
        <p>Audio remains request-scoped unless retention is explicitly selected with consent.</p>
      </footer>
    </div>
  );
}

export default App;
