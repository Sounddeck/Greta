"""
Logs Router for CPAS5
API endpoints for querying and viewing structured logs
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
import json
import os
from pathlib import Path
from loguru import logger

router = APIRouter(prefix="/api/v1/logs", tags=["Logs"])


class LogLevel(str, Enum):
    """Log level enum"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogEntry(BaseModel):
    """Structured log entry model"""
    timestamp: str
    level: str
    message: str
    module: Optional[str] = None
    function: Optional[str] = None
    line: Optional[int] = None
    extra: Optional[Dict[str, Any]] = None


class LogQueryResponse(BaseModel):
    """Log query response"""
    logs: List[LogEntry]
    total: int
    page: int
    page_size: int
    has_more: bool


class LogStats(BaseModel):
    """Log statistics"""
    total_logs: int
    by_level: Dict[str, int]
    by_event: Dict[str, int]
    time_range: Dict[str, Any]  # Changed from Dict[str, str] to allow int for hours


def parse_log_line(line: str) -> Optional[LogEntry]:
    """
    Parse a JSON log line into a LogEntry

    Args:
        line: JSON log line

    Returns:
        LogEntry or None if parsing fails
    """
    try:
        data = json.loads(line)

        # Extract standard fields
        timestamp = data.get("record", {}).get("time", {}).get("repr", "")
        level = data.get("record", {}).get("level", {}).get("name", "INFO")
        message = data.get("text", "").strip()
        module = data.get("record", {}).get("name", "")
        function = data.get("record", {}).get("function", "")
        line_num = data.get("record", {}).get("line", 0)
        extra = data.get("record", {}).get("extra", {})

        # Flatten nested extra.extra to extra (for compatibility with structured logger)
        if "extra" in extra and isinstance(extra["extra"], dict):
            # Merge nested extra into top-level extra
            nested_extra = extra.pop("extra")
            extra.update(nested_extra)

        return LogEntry(
            timestamp=timestamp,
            level=level,
            message=message,
            module=module,
            function=function,
            line=line_num,
            extra=extra
        )
    except Exception as e:
        logger.debug(f"Failed to parse log line: {e}")
        return None


def read_log_file(
    log_file_path: str,
    level: Optional[LogLevel] = None,
    event: Optional[str] = None,
    search: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 100,
    offset: int = 0
) -> List[LogEntry]:
    """
    Read and filter log entries from log file
    
    Args:
        log_file_path: Path to log file
        level: Filter by log level
        event: Filter by event type (from extra.event)
        search: Search in message text
        start_time: Filter logs after this time
        end_time: Filter logs before this time
        limit: Maximum number of logs to return
        offset: Number of logs to skip
        
    Returns:
        List of LogEntry objects
    """
    log_path = Path(log_file_path)
    
    if not log_path.exists():
        return []
    
    logs = []
    skipped = 0
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                
                entry = parse_log_line(line)
                if not entry:
                    continue
                
                # Apply filters
                if level and entry.level != level.value:
                    continue
                
                if event and entry.extra and entry.extra.get("event") != event:
                    continue
                
                if search and search.lower() not in entry.message.lower():
                    continue
                
                if start_time:
                    try:
                        log_time = datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00"))
                        if log_time < start_time:
                            continue
                    except:
                        pass
                
                if end_time:
                    try:
                        log_time = datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00"))
                        if log_time > end_time:
                            continue
                    except:
                        pass
                
                # Apply offset
                if skipped < offset:
                    skipped += 1
                    continue
                
                logs.append(entry)
                
                # Apply limit
                if len(logs) >= limit:
                    break
        
        # Reverse to show newest first
        logs.reverse()
        
    except Exception as e:
        logger.error(f"Failed to read log file: {e}")
    
    return logs


@router.get("/query", response_model=LogQueryResponse)
async def query_logs(
    level: Optional[LogLevel] = Query(None, description="Filter by log level"),
    event: Optional[str] = Query(None, description="Filter by event type"),
    search: Optional[str] = Query(None, description="Search in message text"),
    start_time: Optional[str] = Query(None, description="Start time (ISO format)"),
    end_time: Optional[str] = Query(None, description="End time (ISO format)"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(100, ge=1, le=1000, description="Page size")
):
    """
    Query structured logs with filtering and pagination
    
    Returns logs in reverse chronological order (newest first)
    """
    try:
        # Get log file path from config
        from config import get_settings
        settings = get_settings()
        log_file_path = settings.log_file_path
        
        # Parse time filters
        start_dt = None
        end_dt = None
        
        if start_time:
            try:
                start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            except:
                raise HTTPException(400, "Invalid start_time format. Use ISO format.")
        
        if end_time:
            try:
                end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
            except:
                raise HTTPException(400, "Invalid end_time format. Use ISO format.")
        
        # Calculate offset
        offset = (page - 1) * page_size
        
        # Read logs with filters
        logs = read_log_file(
            log_file_path=log_file_path,
            level=level,
            event=event,
            search=search,
            start_time=start_dt,
            end_time=end_dt,
            limit=page_size + 1,  # Get one extra to check if there are more
            offset=offset
        )
        
        # Check if there are more logs
        has_more = len(logs) > page_size
        if has_more:
            logs = logs[:page_size]
        
        return LogQueryResponse(
            logs=logs,
            total=len(logs),  # Note: This is not the total count, just current page
            page=page,
            page_size=page_size,
            has_more=has_more
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to query logs: {e}")
        raise HTTPException(500, f"Failed to query logs: {str(e)}")


@router.get("/stats", response_model=LogStats)
async def get_log_stats(
    hours: int = Query(24, ge=1, le=168, description="Number of hours to analyze")
):
    """
    Get log statistics for the specified time period
    
    Returns counts by level and event type
    """
    try:
        from config import get_settings
        settings = get_settings()
        log_file_path = settings.log_file_path
        
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # Read all logs in time range
        logs = read_log_file(
            log_file_path=log_file_path,
            start_time=start_time,
            end_time=end_time,
            limit=100000  # Large limit to get all logs
        )
        
        # Calculate statistics
        by_level = {}
        by_event = {}
        
        for log in logs:
            # Count by level
            by_level[log.level] = by_level.get(log.level, 0) + 1
            
            # Count by event
            if log.extra and "event" in log.extra:
                event = log.extra["event"]
                by_event[event] = by_event.get(event, 0) + 1
        
        return LogStats(
            total_logs=len(logs),
            by_level=by_level,
            by_event=by_event,
            time_range={
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "hours": hours
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get log stats: {e}")
        raise HTTPException(500, f"Failed to get log stats: {str(e)}")


@router.get("/events")
async def get_available_events():
    """
    Get list of available event types from recent logs
    
    Useful for populating filter dropdowns
    """
    try:
        from config import get_settings
        settings = get_settings()
        log_file_path = settings.log_file_path
        
        # Read recent logs
        logs = read_log_file(
            log_file_path=log_file_path,
            limit=1000
        )
        
        # Extract unique events
        events = set()
        for log in logs:
            if log.extra and "event" in log.extra:
                events.add(log.extra["event"])
        
        return {
            "events": sorted(list(events)),
            "count": len(events)
        }
        
    except Exception as e:
        logger.error(f"Failed to get available events: {e}")
        raise HTTPException(500, f"Failed to get available events: {str(e)}")

