"""
Standardized error handling for CLI commands.

Provides consistent error messaging and eliminates silent failures.
"""

import os
import traceback

import typer
from graid.cli.exceptions import (
    CLIError,
    ConfigurationError,
    ProcessingError,
    UploadError,
    ValidationError,
)


class ErrorHandler:
    """Standardized error handling for CLI commands."""

    @staticmethod
    def handle_validation_error(error: ValidationError):
        """Handle validation errors with appropriate user messaging."""
        typer.secho(
            f"❌ Validation Error: {error.message}", fg=typer.colors.RED, bold=True
        )
        typer.echo("💡 Use --help for usage information or check your configuration.")
        raise typer.Exit(error.exit_code)

    @staticmethod
    def handle_configuration_error(error: ConfigurationError):
        """Handle configuration errors with helpful suggestions."""
        typer.secho(
            f"❌ Configuration Error: {error.message}", fg=typer.colors.RED, bold=True
        )
        typer.echo("💡 Check your configuration file format and required parameters.")
        raise typer.Exit(error.exit_code)

    @staticmethod
    def handle_processing_error(error: ProcessingError):
        """Handle processing errors with debugging information."""
        typer.secho(
            f"❌ Processing Error: {error.message}", fg=typer.colors.RED, bold=True
        )
        typer.echo(
            "💡 This usually indicates an issue with the dataset or model configuration."
        )

        # Show traceback in debug mode
        if os.getenv("GRAID_DEBUG_VERBOSE"):
            typer.echo("\n🔍 Debug traceback:")
            traceback.print_exc()
        else:
            typer.echo("💡 Set GRAID_DEBUG_VERBOSE=1 for detailed error information.")

        raise typer.Exit(error.exit_code)

    @staticmethod
    def handle_upload_error(error: UploadError):
        """Handle upload errors with network/authentication hints."""
        typer.secho(f"❌ Upload Error: {error.message}", fg=typer.colors.RED, bold=True)
        typer.echo(
            "💡 Check your HuggingFace Hub authentication and network connection."
        )
        typer.echo("💡 Run 'huggingface-cli login' if not authenticated.")
        raise typer.Exit(error.exit_code)

    @staticmethod
    def handle_unexpected_error(error: Exception):
        """Handle unexpected errors with full debugging information."""
        typer.secho(f"❌ Unexpected Error: {error}", fg=typer.colors.RED, bold=True)
        typer.echo(
            "💡 This is likely a bug. Please report it with the traceback below."
        )
        typer.echo()

        # Always show traceback for unexpected errors
        typer.echo("🔍 Full traceback:")
        traceback.print_exc()

        raise typer.Exit(1)

    @staticmethod
    def handle_cli_error(error: CLIError):
        """Route CLI errors to appropriate handlers."""
        if isinstance(error, ValidationError):
            ErrorHandler.handle_validation_error(error)
        elif isinstance(error, ConfigurationError):
            ErrorHandler.handle_configuration_error(error)
        elif isinstance(error, ProcessingError):
            ErrorHandler.handle_processing_error(error)
        elif isinstance(error, UploadError):
            ErrorHandler.handle_upload_error(error)
        else:
            # Generic CLI error
            typer.secho(f"❌ Error: {error.message}", fg=typer.colors.RED, bold=True)
            raise typer.Exit(error.exit_code)

    @staticmethod
    def safe_operation(operation, error_message: str = "Operation failed"):
        """
        Execute operation safely, converting silent failures to explicit errors.

        This replaces patterns like:
            try:
                some_operation()
            except Exception:
                pass  # Silent failure - BAD!

        With:
            ErrorHandler.safe_operation(some_operation, "Failed to perform operation")
        """
        try:
            return operation()
        except Exception as e:
            # Convert silent failure to explicit error
            typer.secho(f"❌ {error_message}: {e}", fg=typer.colors.RED)
            if os.getenv("GRAID_DEBUG_VERBOSE"):
                traceback.print_exc()
            raise CLIError(f"{error_message}: {e}")

    @staticmethod
    def validate_and_execute(validation_func, operation_func, operation_name: str):
        """
        Execute validation followed by operation with proper error handling.

        This ensures validation errors are caught and handled appropriately
        before attempting the main operation.
        """
        try:
            # Validation phase
            validation_func()

            # Operation phase
            return operation_func()

        except CLIError:
            # Re-raise CLI errors as-is (already have proper context)
            raise
        except Exception as e:
            # Convert unexpected errors
            ErrorHandler.handle_unexpected_error(e)
