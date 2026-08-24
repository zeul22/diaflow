export const PERSISTENCE_MODES = Object.freeze({
  NONE: "none",
  RESULT: "result",
  RESULT_AND_AUDIO: "result_and_audio",
});

const VALID_PERSISTENCE_MODES = new Set(Object.values(PERSISTENCE_MODES));
const PERSISTENCE_MODE_RANK = Object.freeze({
  [PERSISTENCE_MODES.NONE]: 0,
  [PERSISTENCE_MODES.RESULT]: 1,
  [PERSISTENCE_MODES.RESULT_AND_AUDIO]: 2,
});

const HISTORY_ERROR_COPY = {
  401: "You are not authorized to view stored analyses.",
  403: "You are not authorized to view stored analyses.",
  404: "Stored-analysis history is not enabled on this service.",
  503: "Stored-analysis history is temporarily unavailable.",
};

export class PersistenceApiError extends Error {
  constructor(message, { code = "PERSISTENCE_REQUEST_FAILED", status = 0 } = {}) {
    super(message);
    this.name = "PersistenceApiError";
    this.code = code;
    this.status = status;
  }
}

export function normalizePersistenceMode(mode) {
  return VALID_PERSISTENCE_MODES.has(mode) ? mode : PERSISTENCE_MODES.NONE;
}

export function persistenceModeIsAvailable(mode, maximumMode) {
  const normalizedMode = normalizePersistenceMode(mode);
  const normalizedMaximum = normalizePersistenceMode(maximumMode);
  return PERSISTENCE_MODE_RANK[normalizedMode] <= PERSISTENCE_MODE_RANK[normalizedMaximum];
}

export function persistenceHeaders({ mode = PERSISTENCE_MODES.NONE, consentReference = "" } = {}) {
  const normalizedMode = normalizePersistenceMode(mode);
  const headers = { "X-Persistence-Mode": normalizedMode };
  const normalizedReference = String(consentReference || "").trim();
  if (normalizedMode === PERSISTENCE_MODES.RESULT_AND_AUDIO && normalizedReference) {
    headers["X-Consent-Reference"] = normalizedReference;
  }
  return headers;
}

export function persistenceStartFields({ mode = PERSISTENCE_MODES.NONE, consentReference = "" } = {}) {
  const normalizedMode = normalizePersistenceMode(mode);
  const fields = { persistence_mode: normalizedMode };
  const normalizedReference = String(consentReference || "").trim();
  if (normalizedMode === PERSISTENCE_MODES.RESULT_AND_AUDIO && normalizedReference) {
    fields.consent_reference = normalizedReference;
  }
  return fields;
}

async function readJson(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

export async function getPersistenceCapabilities({ signal } = {}) {
  let response;
  try {
    response = await fetch("/api/v1/persistence/capabilities", {
      cache: "no-store",
      headers: { Accept: "application/json" },
      signal,
    });
  } catch (error) {
    if (error?.name === "AbortError") throw error;
    throw new PersistenceApiError("Storage capabilities could not be reached.", {
      code: "CAPABILITIES_NETWORK_ERROR",
    });
  }

  const payload = await readJson(response);
  if (!response.ok) {
    throw historyError(response, payload, "Storage capabilities could not be loaded.");
  }
  if (
    !payload ||
    typeof payload !== "object" ||
    Array.isArray(payload) ||
    typeof payload.enabled !== "boolean" ||
    !VALID_PERSISTENCE_MODES.has(payload.maximum_mode)
  ) {
    throw new PersistenceApiError("The service returned invalid storage capabilities.", {
      code: "INVALID_CAPABILITIES_RESPONSE",
      status: response.status,
    });
  }

  return {
    ...payload,
    maximum_mode: payload.enabled ? payload.maximum_mode : PERSISTENCE_MODES.NONE,
    default_mode: PERSISTENCE_MODES.NONE,
  };
}

function historyError(response, payload, fallback) {
  const detail = payload?.error;
  return new PersistenceApiError(
    HISTORY_ERROR_COPY[response.status] || detail?.message || fallback,
    {
      code: detail?.code || `HTTP_${response.status}`,
      status: response.status,
    },
  );
}

export async function listStoredAnalyses({ signal } = {}) {
  let response;
  try {
    response = await fetch("/api/v1/analyses", {
      cache: "no-store",
      headers: { Accept: "application/json" },
      signal,
    });
  } catch (error) {
    if (error?.name === "AbortError") throw error;
    throw new PersistenceApiError("Stored-analysis history could not be reached.", {
      code: "HISTORY_NETWORK_ERROR",
    });
  }

  const payload = await readJson(response);
  if (!response.ok) {
    throw historyError(response, payload, "Stored-analysis history could not be loaded.");
  }

  const analyses = Array.isArray(payload)
    ? payload
    : Array.isArray(payload?.analyses)
      ? payload.analyses
      : Array.isArray(payload?.items)
        ? payload.items
        : null;
  if (!analyses) {
    throw new PersistenceApiError("The service returned an unexpected history response.", {
      code: "INVALID_HISTORY_RESPONSE",
      status: response.status,
    });
  }
  return analyses.filter((analysis) => analysis && typeof analysis === "object");
}

export async function getStoredAnalysis(analysisId, { signal } = {}) {
  const normalizedId = String(analysisId || "").trim();
  if (!normalizedId) {
    throw new PersistenceApiError("A stored analysis ID is required.", {
      code: "MISSING_ANALYSIS_ID",
    });
  }

  let response;
  try {
    response = await fetch(`/api/v1/analyses/${encodeURIComponent(normalizedId)}`, {
      cache: "no-store",
      headers: { Accept: "application/json" },
      signal,
    });
  } catch (error) {
    if (error?.name === "AbortError") throw error;
    throw new PersistenceApiError("Stored-analysis detail could not be reached.", {
      code: "HISTORY_NETWORK_ERROR",
    });
  }

  const payload = await readJson(response);
  if (!response.ok) {
    throw historyError(response, payload, "Stored-analysis detail could not be loaded.");
  }
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new PersistenceApiError("The service returned an unexpected analysis detail.", {
      code: "INVALID_HISTORY_RESPONSE",
      status: response.status,
    });
  }
  return payload;
}

export async function deleteStoredAnalysis(analysisId, { signal } = {}) {
  const normalizedId = String(analysisId || "").trim();
  if (!normalizedId) {
    throw new PersistenceApiError("A stored analysis ID is required.", {
      code: "MISSING_ANALYSIS_ID",
    });
  }

  let response;
  try {
    response = await fetch(`/api/v1/analyses/${encodeURIComponent(normalizedId)}`, {
      method: "DELETE",
      headers: { Accept: "application/json" },
      signal,
    });
  } catch (error) {
    if (error?.name === "AbortError") throw error;
    throw new PersistenceApiError("The stored analysis could not be deleted.", {
      code: "HISTORY_NETWORK_ERROR",
    });
  }

  if (!response.ok) {
    const payload = await readJson(response);
    throw historyError(response, payload, "The stored analysis could not be deleted.");
  }
}
