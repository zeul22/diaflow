from __future__ import annotations

from dataclasses import dataclass

from app.errors import InvalidRequestError, UnsupportedMediaError

SUPPORTED_ENCODINGS = {
    "auto",
    "wav",
    "pcm_s16le",
    "pcm_s16be",
    "pcm_f32le",
    "mulaw",
    "alaw",
}

CONTENT_TYPE_ENCODINGS = {
    "audio/wav": "wav",
    "audio/wave": "wav",
    "audio/x-wav": "wav",
    "audio/l16": "pcm_s16be",
    "audio/pcmu": "mulaw",
    "audio/mulaw": "mulaw",
    "audio/basic": "mulaw",
    "audio/pcma": "alaw",
    "audio/alaw": "alaw",
}


@dataclass(frozen=True, slots=True)
class SourceSpec:
    encoding: str = "auto"
    sample_rate: int = 16_000
    channels: int = 1
    content_type: str | None = None
    force_degraded: bool = False

    @property
    def is_narrowband(self) -> bool:
        return self.sample_rate <= 8_000 or self.encoding in {"mulaw", "alaw"}

    @property
    def is_quality_limited(self) -> bool:
        return self.is_narrowband or self.force_degraded


def source_spec_from_values(
    *,
    encoding: str | None,
    sample_rate: str | int | None,
    channels: str | int | None,
    content_type: str | None,
) -> SourceSpec:
    media_type = (content_type or "").split(";", 1)[0].strip().lower()
    inferred = CONTENT_TYPE_ENCODINGS.get(media_type, "auto")
    normalized_encoding = (encoding or inferred).strip().lower()
    aliases = {
        "pcm16": "pcm_s16le",
        "s16le": "pcm_s16le",
        "s16be": "pcm_s16be",
        "f32le": "pcm_f32le",
        "pcmu": "mulaw",
        "mu-law": "mulaw",
        "ulaw": "mulaw",
        "pcma": "alaw",
        "a-law": "alaw",
        "container": "auto",
    }
    normalized_encoding = aliases.get(normalized_encoding, normalized_encoding)
    if normalized_encoding not in SUPPORTED_ENCODINGS:
        raise UnsupportedMediaError(
            f"Unsupported audio encoding '{normalized_encoding}'"
        )

    default_rate = 8_000 if normalized_encoding in {"mulaw", "alaw"} else 16_000
    try:
        rate = int(sample_rate) if sample_rate is not None else default_rate
        channel_count = int(channels) if channels is not None else 1
    except (TypeError, ValueError) as exc:
        raise InvalidRequestError(
            "INVALID_AUDIO_METADATA", "sample_rate and channels must be integers"
        ) from exc

    if not 8_000 <= rate <= 96_000:
        raise InvalidRequestError(
            "INVALID_SAMPLE_RATE", "sample_rate must be between 8000 and 96000"
        )
    if channel_count not in {1, 2}:
        raise InvalidRequestError(
            "INVALID_CHANNEL_COUNT", "channels must be either 1 or 2"
        )
    return SourceSpec(
        encoding=normalized_encoding,
        sample_rate=rate,
        channels=channel_count,
        content_type=media_type or None,
    )
