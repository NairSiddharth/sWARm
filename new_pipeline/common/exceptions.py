"""
Custom exceptions for the oWAR pipeline.

All pipeline-specific exceptions defined here for consistent error handling.
"""


class OWARPipelineError(Exception):
    """Base exception for all oWAR pipeline errors."""
    pass


class DataValidationError(OWARPipelineError):
    """Raised when data fails validation checks."""
    pass


class MissingColumnError(DataValidationError):
    """Raised when a required column is missing from DataFrame."""

    def __init__(self, missing_columns, available_columns=None):
        self.missing_columns = missing_columns
        self.available_columns = available_columns

        msg = f"Missing required columns: {missing_columns}"
        if available_columns:
            msg += f"\nAvailable columns: {available_columns}"

        super().__init__(msg)


class InvalidDataTypeError(DataValidationError):
    """Raised when column has incorrect data type."""

    def __init__(self, column, expected_type, actual_type):
        self.column = column
        self.expected_type = expected_type
        self.actual_type = actual_type

        super().__init__(
            f"Column '{column}' has invalid type. "
            f"Expected {expected_type}, got {actual_type}"
        )


class DataRangeError(DataValidationError):
    """Raised when data values are outside expected range."""

    def __init__(self, column, expected_range, actual_range):
        self.column = column
        self.expected_range = expected_range
        self.actual_range = actual_range

        super().__init__(
            f"Column '{column}' values outside expected range. "
            f"Expected {expected_range}, found {actual_range}"
        )


class InsufficientDataError(OWARPipelineError):
    """Raised when dataset has too few records for reliable processing."""

    def __init__(self, min_required, actual):
        self.min_required = min_required
        self.actual = actual

        super().__init__(
            f"Insufficient data for processing. "
            f"Required: {min_required}, Actual: {actual}"
        )


class FeatureLoadError(OWARPipelineError):
    """Raised when feature loading fails."""

    def __init__(self, feature_name, reason):
        self.feature_name = feature_name
        self.reason = reason

        super().__init__(
            f"Failed to load feature '{feature_name}': {reason}"
        )


class ImputerNotFittedError(OWARPipelineError):
    """Raised when transform is called before fit on imputer."""

    def __init__(self):
        super().__init__(
            "MissingValueImputer must be fitted before transform. "
            "Call fit() first."
        )
