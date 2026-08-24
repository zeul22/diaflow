CREATE TABLE IF NOT EXISTS persistence_schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS persistence_sessions (
    session_id UUID PRIMARY KEY,
    contact_id UUID NOT NULL,
    mode TEXT NOT NULL CHECK (mode IN ('result', 'result_and_audio')),
    transport TEXT NOT NULL CHECK (transport IN ('rest', 'websocket')),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'completed', 'failed')),
    request_id TEXT,
    content_type TEXT,
    encoding TEXT,
    sample_rate INTEGER CHECK (sample_rate IS NULL OR sample_rate > 0),
    channels INTEGER CHECK (channels IS NULL OR channels > 0),
    model_name TEXT,
    result JSONB,
    error_code TEXT,
    error_message TEXT,
    segment_count INTEGER NOT NULL DEFAULT 0 CHECK (segment_count >= 0),
    audio_bytes BIGINT NOT NULL DEFAULT 0 CHECK (audio_bytes >= 0),
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ NOT NULL,
    CHECK (
        (status = 'pending' AND completed_at IS NULL) OR
        (status IN ('completed', 'failed') AND completed_at IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS persistence_sessions_contact_created_idx
    ON persistence_sessions (contact_id, created_at DESC);

CREATE INDEX IF NOT EXISTS persistence_sessions_expiry_idx
    ON persistence_sessions (expires_at, session_id);

CREATE TABLE IF NOT EXISTS persistence_audio_segments (
    segment_id UUID PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES persistence_sessions(session_id)
        ON DELETE CASCADE,
    sequence INTEGER NOT NULL CHECK (sequence >= 0),
    bucket TEXT NOT NULL,
    object_key TEXT NOT NULL UNIQUE,
    byte_start BIGINT NOT NULL CHECK (byte_start >= 0),
    byte_end BIGINT NOT NULL CHECK (byte_end > byte_start),
    byte_size BIGINT NOT NULL CHECK (byte_size > 0),
    sha256 CHAR(64) NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    content_type TEXT NOT NULL,
    logical_chunks JSONB NOT NULL DEFAULT '[]'::JSONB,
    created_at TIMESTAMPTZ NOT NULL,
    UNIQUE (session_id, sequence),
    CHECK (byte_end - byte_start = byte_size),
    CHECK (jsonb_typeof(logical_chunks) = 'array')
);

CREATE INDEX IF NOT EXISTS persistence_audio_segments_session_idx
    ON persistence_audio_segments (session_id, sequence);

INSERT INTO persistence_schema_migrations (version)
VALUES ('001_initial')
ON CONFLICT (version) DO NOTHING;
