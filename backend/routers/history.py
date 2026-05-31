"""
═══════════════════════════════════
📄 FILE 17/42: backend/routers/history.py
═══════════════════════════════════

BrailleVision AI — History & Settings API Router
Endpoints for scan history CRUD, aggregate usage statistics, and user settings.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query  # type: ignore
from pydantic import BaseModel, Field  # type: ignore

from database import db  # type: ignore

# ─────────────────────────────────────────────────────────────
# CONSTANTS & SETUP
# ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/history", tags=["history"])


# ─────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────


class ScanRecordResponse(BaseModel):
    """Pydantic model representing a ScanRecord."""
    id: int
    raw_text: str
    corrected_text: str
    translated_text: Optional[str] = None
    target_language: Optional[str] = None
    avg_confidence: float
    cell_count: int
    source_type: str
    correction_method: Optional[str] = None
    side_detected: Optional[str] = None
    processing_time_ms: Optional[float] = None
    created_at: Optional[str] = None
    audio_path: Optional[str] = None


class StatsResponse(BaseModel):
    """Pydantic model representing scan statistics."""
    total_scans: int
    total_words: int
    avg_confidence: float
    scans_today: int


class SettingRequest(BaseModel):
    """Request model for upserting an app setting."""
    key: str = Field(..., description="Unique setting identifier")
    value: str = Field(..., description="Value to store")


class SettingResponse(BaseModel):
    """Response model for retrieving or setting an app setting."""
    key: str
    value: str
    updated_at: Optional[str] = None


class DeleteCountResponse(BaseModel):
    """Response returned when records are deleted."""
    success: bool
    deleted_count: int
    message: str


# ─────────────────────────────────────────────────────────────
# HISTORIC SCAN ENDPOINTS
# ─────────────────────────────────────────────────────────────


@router.get("/", response_model=list[ScanRecordResponse], summary="Retrieve scan history (paginated)")
async def get_history(
    limit: int = Query(50, ge=1, le=100, description="Max scan records to return"),
    offset: int = Query(0, ge=0, description="Records to skip"),
) -> list[ScanRecordResponse]:
    """
    Retrieve scan records ordered by creation date (newest first).
    """
    try:
        records = await db.get_all_scans(limit=limit, offset=offset)
        return [ScanRecordResponse(**r.to_dict()) for r in records]
    except Exception as exc:
        logger.error("get_history failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Database query failed: {exc}") from exc


@router.get("/stats", response_model=StatsResponse, summary="Get aggregate scan statistics")
async def get_statistics() -> StatsResponse:
    """
    Retrieve statistics including total scans, total word count, average confidence, and scans today.
    """
    try:
        stats = await db.get_stats()
        return StatsResponse(**stats)
    except Exception as exc:
        logger.error("get_statistics failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to calculate stats: {exc}") from exc


@router.get("/{scan_id}", response_model=ScanRecordResponse, summary="Retrieve a single scan by ID")
async def get_scan(scan_id: int) -> ScanRecordResponse:
    """
    Retrieve a specific scan record by its unique database ID.
    """
    try:
        record = await db.get_scan_by_id(scan_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Scan record with ID {scan_id} not found.")
        return ScanRecordResponse(**record.to_dict())
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("get_scan failed for id=%d: %s", scan_id, exc)
        raise HTTPException(status_code=500, detail=f"Database query failed: {exc}") from exc


@router.delete("/{scan_id}", response_model=DeleteCountResponse, summary="Delete a single scan by ID")
async def delete_scan(scan_id: int) -> DeleteCountResponse:
    """
    Delete a specific scan record from history.
    """
    try:
        deleted = await db.delete_scan(scan_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Scan record with ID {scan_id} not found.")
        return DeleteCountResponse(
            success=True,
            deleted_count=1,
            message=f"Scan record {scan_id} successfully deleted."
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("delete_scan failed for id=%d: %s", scan_id, exc)
        raise HTTPException(status_code=500, detail=f"Failed to delete record: {exc}") from exc


@router.delete("/", response_model=DeleteCountResponse, summary="Clear all scan history")
async def clear_all_history() -> DeleteCountResponse:
    """
    Permanently delete all scan records from the database.
    """
    try:
        count = await db.delete_all_scans()
        return DeleteCountResponse(
            success=True,
            deleted_count=count,
            message=f"All {count} scan records have been deleted."
        )
    except Exception as exc:
        logger.error("clear_all_history failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {exc}") from exc


# ─────────────────────────────────────────────────────────────
# SETTINGS ENDPOINTS
# ─────────────────────────────────────────────────────────────


@router.get("/settings/{key}", response_model=SettingResponse, summary="Get application setting by key")
async def get_setting(key: str, default: Optional[str] = None) -> SettingResponse:
    """
    Retrieve user preferences or system configuration settings.
    """
    try:
        val = await db.get_setting(key, default)
        if val is None:
            raise HTTPException(status_code=404, detail=f"Setting with key '{key}' not found.")
        return SettingResponse(key=key, value=val)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("get_setting failed for key=%s: %s", key, exc)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve setting: {exc}") from exc


@router.post("/settings", response_model=SettingResponse, summary="Upsert application setting")
async def save_setting(request: SettingRequest) -> SettingResponse:
    """
    Save or update a system setting/user preference.
    """
    try:
        setting = await db.save_setting(request.key, request.value)
        return SettingResponse(**setting.to_dict())
    except Exception as exc:
        logger.error("save_setting failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to save setting: {exc}") from exc
