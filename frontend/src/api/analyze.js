import { PERSISTENCE_MODES, persistenceHeaders } from "./persistence.js";

export const MAX_AUDIO_BYTES = 12 * 1024 * 1024;

const AUDIO_EXTENSIONS = new Set([
  "aac",
  "flac",
  "m4a",
  "mp3",
  "mp4",
  "ogg",
  "opus",
  "wav",
  "webm",
]);
const AGE_BRACKETS = new Set(["18-30", "31-45", "46-60", "60+", "unknown"]);
const GENDER_LABELS = new Set(["female", "male", "unknown"]);
const QUALITY_LABELS = new Set(["degraded", "good", "insufficient"]);
const LANGUAGE_CODE_PATTERN = /^(unknown|[a-z]{2,8})$/;

// The service returns an ISO-639 style tag from a 107-language model. Naming
// every one of them here would be noise, so common tags get a readable name and
// anything else falls back to the tag itself rather than showing nothing.
const LANGUAGE_NAMES = {
  ar: "Arabic",
  bn: "Bengali",
  cs: "Czech",
  de: "German",
  el: "Greek",
  en: "English",
  es: "Spanish",
  fa: "Persian",
  fi: "Finnish",
  fr: "French",
  gu: "Gujarati",
  he: "Hebrew",
  hi: "Hindi",
  hu: "Hungarian",
  id: "Indonesian",
  it: "Italian",
  ja: "Japanese",
  kn: "Kannada",
  ko: "Korean",
  ml: "Malayalam",
  mr: "Marathi",
  ms: "Malay",
  nl: "Dutch",
  no: "Norwegian",
  pa: "Punjabi",
  pl: "Polish",
  pt: "Portuguese",
  ro: "Romanian",
  ru: "Russian",
  sv: "Swedish",
  ta: "Tamil",
  te: "Telugu",
  th: "Thai",
  tl: "Tagalog",
  tr: "Turkish",
  uk: "Ukrainian",
  ur: "Urdu",
  vi: "Vietnamese",
  zh: "Chinese",
};

export function languageName(code) {
  if (typeof code !== "string" || !code) return "—";
  if (code === "unknown") return "Not determined";
  return LANGUAGE_NAMES[code] || code.toUpperCase();
}

const FRIENDLY_ERRORS = {
  INPUT_TIMEOUT: "The upload timed out. Check the connection and try again.",
  INVALID_AUDIO: "The recording could not be decoded. Try another audio file.",
  INVALID_CONTACT_ID: "Contact ID must be a valid UUID, or you can leave it blank.",
  MODEL_UNAVAILABLE: "The analyzer is still starting. Try again in a moment.",
  PAYLOAD_TOO_LARGE: "Choose an audio file smaller than 12 MB.",
  SERVICE_BUSY: "The analyzer is busy. Your file is still selected—please retry.",
  UNSUPPORTED_MEDIA_TYPE: "Choose a supported audio recording such as M4A, WAV, or MP3.",
};

export class AnalysisApiError extends Error {
  constructor(message, { code = "REQUEST_FAILED", requestId = null, status = 0 } = {}) {
    super(message);
    this.name = "AnalysisApiError";
    this.code = code;
    this.requestId = requestId;
    this.status = status;
  }
}

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function extensionOf(filename) {
  const parts = filename.toLowerCase().split(".");
  return parts.length > 1 ? parts.at(-1) : "";
}

export function validateAudioFile(file) {
  if (!file || typeof file.size !== "number") {
    return "Choose an audio file to analyze.";
  }
  if (file.size === 0) return "The selected file is empty.";
  if (file.size > MAX_AUDIO_BYTES) return "Choose an audio file smaller than 12 MB.";

  const mediaType = String(file.type || "").toLowerCase();
  const extension = extensionOf(file.name || "");
  if (!mediaType.startsWith("audio/") && !AUDIO_EXTENSIONS.has(extension)) {
    return "Choose a supported audio recording such as M4A, WAV, or MP3.";
  }
  return null;
}

function isPrediction(value, allowedLabels) {
  return Boolean(
    value &&
      typeof value === "object" &&
      allowedLabels.has(value.prediction) &&
      Number.isFinite(value.confidence) &&
      value.confidence >= 0 &&
      value.confidence <= 1,
  );
}

export function isLanguagePrediction(value) {
  return Boolean(
    value &&
      typeof value === "object" &&
      typeof value.prediction === "string" &&
      LANGUAGE_CODE_PATTERN.test(value.prediction) &&
      Number.isFinite(value.confidence) &&
      value.confidence >= 0 &&
      value.confidence <= 1,
  );
}

export function isAnalysisResponse(value) {
  return Boolean(
    value &&
      typeof value === "object" &&
      typeof value.contact_id === "string" &&
      isPrediction(value.gender, GENDER_LABELS) &&
      isPrediction(value.age_bracket, AGE_BRACKETS) &&
      Number.isFinite(value.processing_ms) &&
      value.processing_ms >= 0 &&
      QUALITY_LABELS.has(value.audio_quality) &&
      // Language is optional: deployments can run without the second encoder.
      (value.language === undefined ||
        value.language === null ||
        isLanguagePrediction(value.language)),
  );
}

export async function analyzeAudio({
  file,
  contactId = "",
  persistenceMode = PERSISTENCE_MODES.NONE,
  consentReference = "",
  signal,
}) {
  const form = new FormData();
  form.append("audio", file, file.name || "recording");
  if (contactId.trim()) form.append("contact_id", contactId.trim());

  let response;
  try {
    response = await fetch("/api/v1/analyze", {
      method: "POST",
      body: form,
      signal,
      headers: {
        Accept: "application/json",
        ...persistenceHeaders({ mode: persistenceMode, consentReference }),
      },
    });
  } catch (error) {
    if (error?.name === "AbortError") throw error;
    throw new AnalysisApiError(
      "The service could not be reached. Verify Docker Compose is running.",
      { code: "NETWORK_ERROR" },
    );
  }

  let payload = null;
  try {
    payload = await response.json();
  } catch {
    // Handled below as a malformed success or a generic server error.
  }

  if (!response.ok) {
    const detail = payload?.error;
    const code = detail?.code || `HTTP_${response.status}`;
    throw new AnalysisApiError(
      FRIENDLY_ERRORS[code] || detail?.message || "The analysis request failed.",
      {
        code,
        requestId: detail?.request_id || response.headers.get("X-Request-ID"),
        status: response.status,
      },
    );
  }

  if (!isAnalysisResponse(payload)) {
    throw new AnalysisApiError("The service returned an unexpected response.", {
      code: "INVALID_RESPONSE",
      requestId: response.headers.get("X-Request-ID"),
      status: response.status,
    });
  }
  return payload;
}
