"""
CLI-specific exceptions for better error handling and user messaging.
"""


class CLIError(Exception):
    """Base exception for CLI-related errors."""
    
    def __init__(self, message: str, exit_code: int = 1):
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code


class ValidationError(CLIError):
    """Base exception for validation errors."""
    pass


class DatasetValidationError(ValidationError):
    """Dataset name validation error."""
    pass


class COCOValidationError(ValidationError):
    """COCO object validation error."""
    pass


class SplitValidationError(ValidationError):
    """Split specification validation error."""
    pass


class ConfigurationError(CLIError):
    """Configuration loading/parsing errors."""
    pass


class ProcessingError(CLIError):
    """Dataset processing errors."""
    pass


class UploadError(CLIError):
    """HuggingFace Hub upload errors."""
    pass
