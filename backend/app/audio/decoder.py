from __future__ import annotations

import io
import re
import subprocess
import wave
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from app.audio.types import SourceSpec
from app.config import Settings
from app.errors import InvalidAudioError


@dataclass(frozen=True, slots=True)
class DecodedAudio:
    samples: npt.NDArray[np.float32]
    sample_rate: int
    source_sample_rate: int
    source_encoding: str
    source_metadata_known: bool
    used_ffmpeg: bool

    @property
    def duration_seconds(self) -> float:
        return float(self.samples.size / self.sample_rate)


class AudioDecoder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def decode(self, payload: bytes | bytearray, source: SourceSpec) -> DecodedAudio:
        if not payload:
            raise InvalidAudioError("The audio payload is empty")

        encoding = source.encoding
        source_sample_rate = source.sample_rate
        source_encoding = encoding
        source_metadata_known = encoding != "auto"
        used_ffmpeg = False
        if encoding == "wav" or self._looks_like_wav(payload):
            try:
                samples, sample_rate = self._decode_wav(payload)
                source_sample_rate = sample_rate
                source_encoding = "wav"
                source_metadata_known = True
            except (EOFError, ValueError, wave.Error):
                (
                    samples,
                    sample_rate,
                    probed_encoding,
                    probed_sample_rate,
                ) = self._decode_with_ffmpeg(payload)
                if probed_encoding is not None:
                    source_encoding = probed_encoding
                if probed_sample_rate is not None:
                    source_sample_rate = probed_sample_rate
                source_metadata_known = (
                    probed_encoding is not None and probed_sample_rate is not None
                )
                used_ffmpeg = True
        elif encoding in {"pcm_s16le", "pcm_s16be", "pcm_f32le"}:
            samples, sample_rate = self._decode_pcm(payload, source)
        elif encoding == "mulaw":
            samples = self._mix_channels(self._decode_mulaw(payload), source.channels)
            sample_rate = source.sample_rate
        elif encoding == "alaw":
            samples = self._mix_channels(self._decode_alaw(payload), source.channels)
            sample_rate = source.sample_rate
        else:
            (
                samples,
                sample_rate,
                probed_encoding,
                probed_sample_rate,
            ) = self._decode_with_ffmpeg(payload)
            if probed_encoding is not None:
                source_encoding = probed_encoding
            if probed_sample_rate is not None:
                source_sample_rate = probed_sample_rate
            source_metadata_known = (
                probed_encoding is not None and probed_sample_rate is not None
            )
            used_ffmpeg = True

        if not 8_000 <= sample_rate <= 96_000:
            raise InvalidAudioError("Audio sample rate must be between 8000 and 96000")
        source_duration = samples.size / sample_rate
        if source_duration > self.settings.max_audio_seconds + 0.05:
            # Reject before allocating the potentially much larger resampled
            # arrays. This also bounds deliberately malformed WAV metadata.
            raise InvalidAudioError(
                f"Decoded audio exceeds the {self.settings.max_audio_seconds:g}-second limit"
            )

        if sample_rate != self.settings.target_sample_rate:
            samples = self._resample_linear(
                samples, sample_rate, self.settings.target_sample_rate
            )
            sample_rate = self.settings.target_sample_rate

        samples = np.nan_to_num(
            np.asarray(samples, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0
        )
        samples = np.clip(samples, -1.0, 1.0).astype(np.float32, copy=False)
        duration = samples.size / sample_rate
        if duration > self.settings.max_audio_seconds + 0.05:
            raise InvalidAudioError(
                f"Decoded audio exceeds the {self.settings.max_audio_seconds:g}-second limit"
            )
        if samples.size == 0:
            raise InvalidAudioError("The audio payload contains no samples")
        return DecodedAudio(
            samples=samples,
            sample_rate=sample_rate,
            source_sample_rate=source_sample_rate,
            source_encoding=source_encoding,
            source_metadata_known=source_metadata_known,
            used_ffmpeg=used_ffmpeg,
        )

    @staticmethod
    def _looks_like_wav(payload: bytes | bytearray) -> bool:
        return (
            len(payload) >= 12
            and bytes(payload[:4]) in {b"RIFF", b"RIFX"}
            and bytes(payload[8:12]) == b"WAVE"
        )

    @staticmethod
    def _decode_wav(
        payload: bytes | bytearray,
    ) -> tuple[npt.NDArray[np.float32], int]:
        with wave.open(io.BytesIO(payload), "rb") as audio_file:
            channels = audio_file.getnchannels()
            sample_width = audio_file.getsampwidth()
            sample_rate = audio_file.getframerate()
            frames = audio_file.readframes(audio_file.getnframes())

        if channels not in {1, 2} or sample_rate <= 0:
            raise ValueError("unsupported WAV channel count or sample rate")
        if sample_width == 1:
            samples = (
                np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0
            ) / 128.0
        elif sample_width == 2:
            samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        elif sample_width == 3:
            packed = np.frombuffer(frames, dtype=np.uint8)
            if packed.size % 3:
                raise ValueError("misaligned 24-bit WAV")
            triples = packed.reshape(-1, 3).astype(np.int32)
            values = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
            values = values - ((values & 0x800000) << 1)
            samples = values.astype(np.float32) / 8388608.0
        elif sample_width == 4:
            samples = (
                np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
            )
        else:
            raise ValueError("unsupported WAV sample width")
        return AudioDecoder._mix_channels(samples, channels), sample_rate

    @staticmethod
    def _decode_pcm(
        payload: bytes | bytearray, source: SourceSpec
    ) -> tuple[npt.NDArray[np.float32], int]:
        if source.encoding == "pcm_f32le":
            width = 4
            dtype = "<f4"
            scale = 1.0
        else:
            width = 2
            dtype = "<i2" if source.encoding == "pcm_s16le" else ">i2"
            scale = 32768.0
        alignment = width * source.channels
        if len(payload) % alignment:
            raise InvalidAudioError("Raw PCM payload is not sample-aligned")
        samples = np.frombuffer(payload, dtype=dtype).astype(np.float32) / scale
        return AudioDecoder._mix_channels(samples, source.channels), source.sample_rate

    @staticmethod
    def _mix_channels(
        samples: npt.NDArray[np.float32], channels: int
    ) -> npt.NDArray[np.float32]:
        if channels == 1:
            return np.asarray(samples, dtype=np.float32)
        if samples.size % channels:
            raise InvalidAudioError("Audio payload is not channel-aligned")
        return samples.reshape(-1, channels).mean(axis=1, dtype=np.float32)

    @staticmethod
    def _decode_mulaw(payload: bytes | bytearray) -> npt.NDArray[np.float32]:
        values = np.frombuffer(payload, dtype=np.uint8)
        decoded = np.bitwise_not(values)
        sign = decoded & 0x80
        exponent = (decoded >> 4) & 0x07
        mantissa = decoded & 0x0F
        magnitude = (((mantissa.astype(np.int32) << 3) + 0x84) << exponent) - 0x84
        signed = np.where(sign != 0, -magnitude, magnitude)
        return (signed.astype(np.float32) / 32768.0).astype(np.float32)

    @staticmethod
    def _decode_alaw(payload: bytes | bytearray) -> npt.NDArray[np.float32]:
        encoded = np.frombuffer(payload, dtype=np.uint8) ^ 0x55
        segment = (encoded & 0x70) >> 4
        magnitude = (encoded & 0x0F).astype(np.int32) << 4
        magnitude = np.where(segment == 0, magnitude + 8, magnitude + 0x108)
        # Cast before subtracting: ``segment`` is uint8 and segment zero would
        # otherwise wrap to 255 even though that branch is not selected.
        shift = np.maximum(segment.astype(np.int32) - 1, 0)
        magnitude = np.where(segment > 1, magnitude << shift, magnitude)
        signed = np.where((encoded & 0x80) != 0, magnitude, -magnitude)
        return (signed.astype(np.float32) / 32768.0).astype(np.float32)

    @staticmethod
    def _resample_linear(
        samples: npt.NDArray[np.float32], source_rate: int, target_rate: int
    ) -> npt.NDArray[np.float32]:
        if samples.size < 2:
            return samples.astype(np.float32, copy=False)
        output_size = max(1, int(round(samples.size * target_rate / source_rate)))
        source_positions = np.arange(samples.size, dtype=np.float64)
        target_positions = np.arange(output_size, dtype=np.float64) * (
            source_rate / target_rate
        )
        return np.interp(target_positions, source_positions, samples).astype(np.float32)

    def _decode_with_ffmpeg(
        self, payload: bytes | bytearray
    ) -> tuple[npt.NDArray[np.float32], int, str | None, int | None]:
        duration_limit = self.settings.max_audio_seconds + 0.1
        command = [
            self.settings.ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "info",
            "-nostdin",
            "-threads",
            "1",
            "-probesize",
            "1048576",
            "-analyzeduration",
            "2000000",
            "-i",
            "pipe:0",
            "-vn",
            "-map_metadata",
            "-1",
            "-ac",
            "1",
            "-ar",
            str(self.settings.target_sample_rate),
            "-t",
            f"{duration_limit:.3f}",
            "-f",
            "f32le",
            "pipe:1",
        ]
        try:
            completed = subprocess.run(
                command,
                input=payload,
                capture_output=True,
                check=False,
                timeout=self.settings.decode_timeout_seconds,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise InvalidAudioError() from exc
        if completed.returncode != 0 or not completed.stdout:
            raise InvalidAudioError()
        if len(completed.stdout) % 4:
            raise InvalidAudioError()
        maximum_output = int(
            (self.settings.max_audio_seconds + 0.2)
            * self.settings.target_sample_rate
            * 4
        )
        if len(completed.stdout) > maximum_output:
            raise InvalidAudioError("Decoded audio exceeds the duration limit")
        source_encoding, source_sample_rate = self._parse_ffmpeg_audio_info(
            completed.stderr
        )
        return (
            np.frombuffer(completed.stdout, dtype="<f4").astype(np.float32),
            self.settings.target_sample_rate,
            source_encoding,
            source_sample_rate,
        )

    @staticmethod
    def _parse_ffmpeg_audio_info(stderr: bytes) -> tuple[str | None, int | None]:
        """Read source stream facts from FFmpeg's bounded decode diagnostics.

        FFmpeg prints the input stream before the generated PCM output stream, so
        selecting the first match avoids mistaking the forced 16 kHz output for
        the source. No second probe process or persistence of the input is needed.
        """

        diagnostics = stderr.decode("utf-8", errors="replace")
        match = re.search(
            r"Audio:\s*([A-Za-z0-9_]+)[^\r\n]*?,\s*(\d{4,6})\s+Hz\b",
            diagnostics,
        )
        if match is None:
            return None, None
        sample_rate = int(match.group(2))
        if not 8_000 <= sample_rate <= 96_000:
            return None, None
        return match.group(1).lower(), sample_rate
