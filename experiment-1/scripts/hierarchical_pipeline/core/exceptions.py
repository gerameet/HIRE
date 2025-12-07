"""Custom exceptions for hierarchical pipeline.

Provides specific exception types for better error handling and debugging.
"""


class PipelineError(Exception):
    """Base exception for all pipeline errors."""
    pass


class ModelNotFoundError(PipelineError):
    """Model checkpoint or weights not found."""
    
    def __init__(self, model_name: str, message: str = None):
        self.model_name = model_name
        if message is None:
            message = (
                f"Model '{model_name}' not found. "
                f"Run './experiment models download {model_name}' to download it."
            )
        super().__init__(message)


class InvalidInputError(PipelineError):
    """Invalid input data provided to pipeline component."""
    pass


class EmbeddingError(PipelineError):
    """Embedding generation failed."""
    pass


class SegmentationError(PipelineError):
    """Segmentation failed."""
    pass


class HierarchyBuildError(PipelineError):
    """Hierarchy construction failed."""
    pass


class EvaluationError(PipelineError):
    """Evaluation task failed."""
    pass


class ConfigurationError(PipelineError):
    """Invalid configuration provided."""
    
    def __init__(self, message: str, config_key: str = None):
        self.config_key = config_key
        if config_key:
            message = f"Configuration error in '{config_key}': {message}"
        super().__init__(message)


class CacheError(PipelineError):
    """Cache operation failed."""
    pass


class ValidationError(InvalidInputError):
    """Input validation failed."""
    pass
