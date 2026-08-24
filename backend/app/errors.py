from __future__ import annotations


class ServiceError(Exception):
    def __init__(self, code: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class RequestTooLargeError(ServiceError):
    def __init__(
        self, message: str = "Audio payload exceeds the configured limit"
    ) -> None:
        super().__init__("PAYLOAD_TOO_LARGE", message, 413)


class UnsupportedMediaError(ServiceError):
    def __init__(
        self, message: str = "Unsupported audio media type or encoding"
    ) -> None:
        super().__init__("UNSUPPORTED_MEDIA_TYPE", message, 415)


class InvalidRequestError(ServiceError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message, 422)


class InvalidAudioError(ServiceError):
    def __init__(self, message: str = "The audio payload could not be decoded") -> None:
        super().__init__("INVALID_AUDIO", message, 422)


class InputTimeoutError(ServiceError):
    def __init__(
        self,
        code: str = "INPUT_TIMEOUT",
        message: str = "Timed out while waiting for audio input",
    ) -> None:
        super().__init__(code, message, 408)


class ServiceBusyError(ServiceError):
    def __init__(self) -> None:
        super().__init__(
            "SERVICE_BUSY",
            "Inference capacity is temporarily full; retry shortly",
            503,
        )


class ModelUnavailableError(ServiceError):
    def __init__(self) -> None:
        super().__init__("MODEL_UNAVAILABLE", "The inference model is not ready", 503)
