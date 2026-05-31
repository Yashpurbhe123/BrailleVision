"""
═══════════════════════════════════
📄 FILE 12/42: backend/database/models.py
═══════════════════════════════════

BrailleVision AI — SQLAlchemy ORM Models
Async-compatible models for scan history and app settings.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (  # type: ignore
    Column,
    DateTime,
    Float,
    Integer,
    String,
)
from sqlalchemy.orm import DeclarativeBase  # type: ignore

# ─────────────────────────────────────────────────────────────
# BASE
# ─────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


# ─────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────


class ScanRecord(Base):
    """
    Stores the result of a single Braille scan operation.

    Each row represents one image processed through the BrailleVision
    AI pipeline, capturing raw text, corrections, translations, and
    quality metrics.
    """

    __tablename__ = "scan_records"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Text content
    raw_text = Column(String, nullable=False, default="")
    corrected_text = Column(String, nullable=False, default="")
    translated_text = Column(String, nullable=True)
    target_language = Column(String, nullable=True)           # e.g. 'hi', 'fr'

    # Quality metrics
    avg_confidence = Column(Float, nullable=False, default=0.0)
    cell_count = Column(Integer, nullable=False, default=0)

    # Scan metadata
    source_type = Column(String, nullable=False, default="image")  # image|live|pdf
    correction_method = Column(String, nullable=True)              # llm|spellcheck|none
    side_detected = Column(String, nullable=True)                  # front|back
    processing_time_ms = Column(Float, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Optional audio export path
    audio_path = Column(String, nullable=True)

    def to_dict(self) -> dict:
        """Serialise record to a JSON-compatible dict."""
        return {
            "id": self.id,
            "raw_text": self.raw_text,
            "corrected_text": self.corrected_text,
            "translated_text": self.translated_text,
            "target_language": self.target_language,
            "avg_confidence": self.avg_confidence,
            "cell_count": self.cell_count,
            "source_type": self.source_type,
            "correction_method": self.correction_method,
            "side_detected": self.side_detected,
            "processing_time_ms": self.processing_time_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "audio_path": self.audio_path,
        }

    def __repr__(self) -> str:
        return (
            f"<ScanRecord id={self.id} "
            f"text='{(self.corrected_text or '')[:20]}...' "
            f"conf={self.avg_confidence:.2f}>"
        )


class AppSetting(Base):
    """
    Key-value store for persistent application settings.

    Used for user preferences that survive app restarts:
    language choice, voice mode, high-contrast mode, etc.
    """

    __tablename__ = "app_settings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String, unique=True, nullable=False, index=True)
    value = Column(String, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    def to_dict(self) -> dict:
        """Serialise setting to dict."""
        return {
            "id": self.id,
            "key": self.key,
            "value": self.value,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self) -> str:
        return f"<AppSetting {self.key}={self.value!r}>"


# ─────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("\n" + "=" * 50)
    print("  ORM Models Smoke Test")
    print("=" * 50)

    # Test ScanRecord
    rec = ScanRecord(
        raw_text="helo wrold",
        corrected_text="hello world",
        avg_confidence=0.85,
        cell_count=10,
        source_type="image",
        correction_method="llm",
        side_detected="front",
        processing_time_ms=342.5,
    )
    d = rec.to_dict()
    print(f"  ScanRecord.to_dict keys: {list(d.keys())}")
    assert "raw_text" in d and "corrected_text" in d
    print(f"  ScanRecord repr: {rec!r}")

    # Test AppSetting
    setting = AppSetting(key="preferred_language", value="hi")
    print(f"  AppSetting repr: {setting!r}")
    sd = setting.to_dict()
    assert sd["key"] == "preferred_language"
    print(f"  AppSetting.to_dict: {sd}")

    print("\n✅ Model smoke test complete.\n")
