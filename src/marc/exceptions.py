class MarcError(Exception):
    """Base exception for MARC configuration and pipeline errors."""


class ConfigError(MarcError):
    """Raised when a pipeline configuration file is missing, malformed, or invalid."""
