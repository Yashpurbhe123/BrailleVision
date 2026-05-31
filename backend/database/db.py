"""
═══════════════════════════════════
📄 FILE 13/42: backend/database/db.py
═══════════════════════════════════

BrailleVision AI — Async Database Operations
SQLAlchemy async session management + full CRUD for scan history
and app settings. Uses aiosqlite for non-blocking SQLite access.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, date
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine  # type: ignore
from sqlalchemy import select, func, delete  # type: ignore

from database.models import Base, ScanRecord, AppSetting  # type: ignore

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL", "sqlite+aiosqlite:///./braillevision.db"
)

DEFAULT_SCAN_LIMIT = 50

# ─────────────────────────────────────────────────────────────
# ENGINE & SESSION
# ─────────────────────────────────────────────────────────────

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ─────────────────────────────────────────────────────────────
# INITIALISATION
# ─────────────────────────────────────────────────────────────


async def init_db() -> None:
    """
    Create all database tables if they do not exist.

    Must be called once at application startup before any DB operations.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("init_db: tables created/verified at '%s'", DATABASE_URL)


# ─────────────────────────────────────────────────────────────
# SCAN RECORD OPERATIONS
# ─────────────────────────────────────────────────────────────


async def save_scan(data: dict) -> ScanRecord:
    """
    Persist a scan result to the database.

    Args:
        data: Dict with scan fields. Recognised keys:
            raw_text, corrected_text, translated_text, target_language,
            avg_confidence, cell_count, source_type, correction_method,
            side_detected, processing_time_ms, audio_path.

    Returns:
        Saved ScanRecord ORM object (with auto-assigned id).
    """
    async with AsyncSessionLocal() as session:
        record = ScanRecord(
            raw_text=data.get("raw_text", ""),
            corrected_text=data.get("corrected_text", ""),
            translated_text=data.get("translated_text"),
            target_language=data.get("target_language"),
            avg_confidence=float(data.get("avg_confidence", 0.0)),
            cell_count=int(data.get("cell_count", 0)),
            source_type=data.get("source_type", "image"),
            correction_method=data.get("correction_method"),
            side_detected=data.get("side_detected"),
            processing_time_ms=data.get("processing_time_ms"),
            audio_path=data.get("audio_path"),
            created_at=datetime.utcnow(),
        )
        session.add(record)
        await session.commit()
        await session.refresh(record)
        logger.info("save_scan: id=%d text='%s...'", record.id, record.corrected_text[:30])
        return record


async def get_all_scans(
    limit: int = DEFAULT_SCAN_LIMIT, offset: int = 0
) -> list[ScanRecord]:
    """
    Retrieve scan records ordered by creation time (newest first).

    Args:
        limit: Maximum number of records to return (default 50).
        offset: Number of records to skip (for pagination).

    Returns:
        List of ScanRecord objects.
    """
    async with AsyncSessionLocal() as session:
        stmt = (
            select(ScanRecord)
            .order_by(ScanRecord.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await session.execute(stmt)
        records = list(result.scalars().all())
        logger.debug("get_all_scans: returned %d records", len(records))
        return records


async def get_scan_by_id(scan_id: int) -> Optional[ScanRecord]:
    """
    Fetch a single scan record by primary key.

    Args:
        scan_id: Record primary key.

    Returns:
        ScanRecord if found, None otherwise.
    """
    async with AsyncSessionLocal() as session:
        record = await session.get(ScanRecord, scan_id)
        if record is None:
            logger.debug("get_scan_by_id: id=%d not found", scan_id)
        return record


async def delete_scan(scan_id: int) -> bool:
    """
    Delete a scan record by primary key.

    Args:
        scan_id: Record primary key.

    Returns:
        True if deleted, False if not found.
    """
    async with AsyncSessionLocal() as session:
        record = await session.get(ScanRecord, scan_id)
        if record is None:
            logger.debug("delete_scan: id=%d not found", scan_id)
            return False
        await session.delete(record)
        await session.commit()
        logger.info("delete_scan: id=%d deleted", scan_id)
        return True


async def delete_all_scans() -> int:
    """
    Delete all scan records.

    Returns:
        Number of records deleted.
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(delete(ScanRecord))
        await session.commit()
        count = result.rowcount
        logger.info("delete_all_scans: deleted %d records", count)
        return count


# ─────────────────────────────────────────────────────────────
# APP SETTINGS OPERATIONS
# ─────────────────────────────────────────────────────────────


async def save_setting(key: str, value: str) -> AppSetting:
    """
    Upsert an application setting.

    Args:
        key: Setting key (unique).
        value: Setting value as string.

    Returns:
        Saved/updated AppSetting object.
    """
    async with AsyncSessionLocal() as session:
        stmt = select(AppSetting).where(AppSetting.key == key)
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            existing.value = value
            existing.updated_at = datetime.utcnow()
            await session.commit()
            await session.refresh(existing)
            logger.debug("save_setting: updated '%s'='%s'", key, value)
            return existing
        else:
            setting = AppSetting(key=key, value=value, updated_at=datetime.utcnow())
            session.add(setting)
            await session.commit()
            await session.refresh(setting)
            logger.debug("save_setting: created '%s'='%s'", key, value)
            return setting


async def get_setting(
    key: str, default: Optional[str] = None
) -> Optional[str]:
    """
    Retrieve an application setting value by key.

    Args:
        key: Setting key.
        default: Value to return if key not found.

    Returns:
        Setting value string, or default if not found.
    """
    async with AsyncSessionLocal() as session:
        stmt = select(AppSetting).where(AppSetting.key == key)
        result = await session.execute(stmt)
        setting = result.scalar_one_or_none()
        if setting is None:
            logger.debug("get_setting: '%s' not found, returning default", key)
            return default
        return setting.value


# ─────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────


async def get_stats() -> dict:
    """
    Compute aggregate usage statistics from scan history.

    Returns:
        Dict with total_scans, total_words, avg_confidence, scans_today.
    """
    async with AsyncSessionLocal() as session:
        # Total scan count
        total_stmt = select(func.count(ScanRecord.id))
        total_result = await session.execute(total_stmt)
        total_scans = total_result.scalar() or 0

        # Average confidence
        conf_stmt = select(func.avg(ScanRecord.avg_confidence))
        conf_result = await session.execute(conf_stmt)
        avg_conf = conf_result.scalar() or 0.0

        # All corrected texts for word count
        text_stmt = select(ScanRecord.corrected_text)
        text_result = await session.execute(text_stmt)
        texts = text_result.scalars().all()
        total_words = sum(len(t.split()) for t in texts if t)

        # Scans today
        today_start = datetime.combine(date.today(), datetime.min.time())
        today_stmt = select(func.count(ScanRecord.id)).where(
            ScanRecord.created_at >= today_start
        )
        today_result = await session.execute(today_stmt)
        scans_today = today_result.scalar() or 0

        stats = {
            "total_scans": total_scans,
            "total_words": total_words,
            "avg_confidence": round(float(avg_conf), 3),
            "scans_today": scans_today,
        }
        logger.debug("get_stats: %s", stats)
        return stats


# ─────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    import logging as _logging
    _logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Override to use in-memory DB for testing
    DATABASE_URL = "sqlite+aiosqlite:///:memory:"
    engine = create_async_engine(DATABASE_URL, echo=False)
    AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    async def run() -> None:
        print("\n" + "=" * 50)
        print("  Database Operations Smoke Test")
        print("=" * 50)

        await init_db()
        print("  ✓ init_db")

        # Save scans
        rec1 = await save_scan({
            "raw_text": "helo", "corrected_text": "hello",
            "avg_confidence": 0.85, "cell_count": 5,
            "source_type": "image", "correction_method": "llm",
            "side_detected": "front", "processing_time_ms": 300.0,
        })
        rec2 = await save_scan({
            "raw_text": "wrold", "corrected_text": "world",
            "avg_confidence": 0.75, "cell_count": 5,
            "source_type": "live",
        })
        print(f"  ✓ save_scan: ids={rec1.id}, {rec2.id}")

        # Get all
        records = await get_all_scans(limit=10)
        assert len(records) == 2
        print(f"  ✓ get_all_scans: {len(records)} records")

        # Get by id
        fetched = await get_scan_by_id(rec1.id)
        assert fetched is not None
        assert fetched.corrected_text == "hello"
        print(f"  ✓ get_scan_by_id: '{fetched.corrected_text}'")

        # Delete one
        deleted = await delete_scan(rec1.id)
        assert deleted is True
        gone = await get_scan_by_id(rec1.id)
        assert gone is None
        print("  ✓ delete_scan")

        # Settings
        await save_setting("language", "hi")
        val = await get_setting("language")
        assert val == "hi"
        await save_setting("language", "fr")  # update
        val2 = await get_setting("language")
        assert val2 == "fr"
        default = await get_setting("nonexistent", default="en")
        assert default == "en"
        print("  ✓ save_setting / get_setting / default")

        # Stats
        stats = await get_stats()
        print(f"  ✓ get_stats: {stats}")

        print("\n✅ Smoke test complete.\n")

    asyncio.run(run())
