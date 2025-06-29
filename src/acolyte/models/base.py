"""
Base models and common mixins.
Provides reusable functionality for all models.
"""

from datetime import datetime
from typing import Optional, Protocol, runtime_checkable, TypeAlias
from pydantic import BaseModel, Field, ConfigDict
import uuid

# Import centralized ID generator
from acolyte.core.id_generator import generate_id
from acolyte.core.utils.datetime_utils import utc_now


class TimestampMixin(BaseModel):
    """
    Mixin for automatic timestamps.
    Adds created_at and updated_at to any model.
    """

    created_at: datetime = Field(default_factory=utc_now, description="UTC creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="UTC last update timestamp")

    def touch(self) -> None:
        """Updates the modification timestamp."""
        self.updated_at = utc_now()


class AcolyteBaseModel(BaseModel):
    """
    Base model for all ACOLYTE.
    Common configuration and enhanced validation.
    """

    model_config = ConfigDict(
        # Validate values on assignment
        validate_assignment=True,
        # Use enum values
        use_enum_values=True,
        # Better JSON serialization
        json_encoders={
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v),
        },
        # Prevent extra fields
        extra="forbid",
        # Better documentation
        json_schema_extra={"additionalProperties": False},
    )


# ============================================================================
# STRATEGY PATTERN FOR IDs - Architectural Migration
# ============================================================================


@runtime_checkable
class Identifiable(Protocol):
    """
    Unified protocol for model identification.

    Allows different ID strategies while maintaining a common interface.
    Used by services for uniform access to identifiers.
    """

    @property
    def primary_key(self) -> str:
        """Returns the primary ID value of the model."""
        ...  # pragma: no cover

    @property
    def primary_key_field(self) -> str:
        """Returns the name of the field acting as PK."""
        ...  # pragma: no cover


class StandardIdMixin(BaseModel):
    """
    Standard strategy with 'id' field.

    For code models, documents and general entities.
    Gradually replaces IdentifiableMixin.
    """

    id: str = Field(
        default_factory=generate_id, description="Unique hex32 identifier (SQLite compatible)"
    )

    @property
    def primary_key(self) -> str:
        """Returns the model ID."""
        return self.id

    @property
    def primary_key_field(self) -> str:
        """Returns 'id' as the PK field name."""
        return "id"


class SessionIdMixin(BaseModel):
    """
    Specialized strategy for conversations with 'session_id'.

    For the conversations domain that uses session_id for business
    reasons and compatibility with associative memory system.
    """

    session_id: str = Field(
        default_factory=generate_id, description="Unique session identifier (hex32)"
    )

    @property
    def primary_key(self) -> str:
        """Returns the model's session_id."""
        return self.session_id

    @property
    def primary_key_field(self) -> str:
        """Returns 'session_id' as the PK field name."""
        return "session_id"


# ============================================================================
# BACKWARDS COMPATIBILITY
# ============================================================================

# Alias for gradual migration without breaking changes
IdentifiableMixin: TypeAlias = StandardIdMixin


# Helper function for services
def get_model_primary_key(model: Identifiable) -> str:
    """
    Gets the primary ID of any model implementing Identifiable.

    Args:
        model: Any model with Identifiable protocol

    Returns:
        Primary ID value

    Example:
        >>> chunk = Chunk(content="code")
        >>> conversation = Conversation()
        >>> get_model_primary_key(chunk)        # chunk.id
        >>> get_model_primary_key(conversation) # conversation.session_id
    """
    return model.primary_key


def get_model_primary_key_field(model: Identifiable) -> str:
    """
    Gets the PK field name of any model.

    Args:
        model: Any model with Identifiable protocol

    Returns:
        Name of the field acting as PK

    Example:
        >>> get_model_primary_key_field(chunk)        # "id"
        >>> get_model_primary_key_field(conversation) # "session_id"
    """
    return model.primary_key_field
