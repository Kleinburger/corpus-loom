class OllamaError(Exception):
    """Base exception for Ollama client errors."""


class JsonExtractionError(OllamaError):
    """Raised when JSON cannot be extracted from model output."""


class ValidationFailedError(OllamaError):
    """Raised when JSON fails schema validation after retries."""
