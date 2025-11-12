"""
System Resources Router for CPAS4
Real-time system resource monitoring
"""
from fastapi import APIRouter
import psutil
import platform
from datetime import datetime
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/system", tags=["System Resources"])


@router.get("/resources")
async def get_system_resources() -> Dict[str, Any]:
    """Get current system resource usage"""
    try:
        # CPU Information
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        # Memory Information
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk Information
        disk = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # Network Information
        net_io = psutil.net_io_counters()
        
        # Process Information
        process_count = len(psutil.pids())
        
        # System Information
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime_seconds = (datetime.now() - boot_time).total_seconds()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "cpu": {
                "percent": round(cpu_percent, 1),
                "count": cpu_count,
                "frequency_mhz": round(cpu_freq.current, 0) if cpu_freq else None,
                "status": get_status_level(cpu_percent)
            },
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "percent": round(memory.percent, 1),
                "status": get_status_level(memory.percent)
            },
            "swap": {
                "total_gb": round(swap.total / (1024**3), 2),
                "used_gb": round(swap.used / (1024**3), 2),
                "percent": round(swap.percent, 1),
                "status": get_status_level(swap.percent)
            },
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percent": round(disk.percent, 1),
                "status": get_status_level(disk.percent)
            },
            "disk_io": {
                "read_mb": round(disk_io.read_bytes / (1024**2), 2) if disk_io else 0,
                "write_mb": round(disk_io.write_bytes / (1024**2), 2) if disk_io else 0,
                "read_count": disk_io.read_count if disk_io else 0,
                "write_count": disk_io.write_count if disk_io else 0
            },
            "network": {
                "sent_mb": round(net_io.bytes_sent / (1024**2), 2),
                "recv_mb": round(net_io.bytes_recv / (1024**2), 2),
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            },
            "system": {
                "platform": platform.system(),
                "platform_release": platform.release(),
                "platform_version": platform.version(),
                "architecture": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "boot_time": boot_time.isoformat(),
                "uptime_hours": round(uptime_seconds / 3600, 1),
                "process_count": process_count
            }
        }
    except Exception as e:
        logger.error(f"Error getting system resources: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/resources/history")
async def get_resource_history(minutes: int = 60) -> Dict[str, Any]:
    """Get historical resource usage (placeholder for future implementation)"""
    return {
        "message": "Historical data collection not yet implemented",
        "requested_minutes": minutes,
        "suggestion": "Use /resources endpoint for current stats"
    }


@router.get("/processes")
async def get_top_processes(limit: int = 10) -> List[Dict[str, Any]]:
    """Get top processes by CPU and memory usage"""
    try:
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
            try:
                pinfo = proc.info
                processes.append({
                    "pid": pinfo['pid'],
                    "name": pinfo['name'],
                    "cpu_percent": round(pinfo['cpu_percent'] or 0, 1),
                    "memory_percent": round(pinfo['memory_percent'] or 0, 1),
                    "status": pinfo['status']
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Sort by CPU usage
        processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
        
        return processes[:limit]
    except Exception as e:
        logger.error(f"Error getting processes: {e}")
        return []


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        
        # Determine overall health
        if cpu_percent > 90 or memory_percent > 90 or disk_percent > 90:
            status = "critical"
        elif cpu_percent > 75 or memory_percent > 75 or disk_percent > 75:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "module": "System Resources",
            "features": [
                "Real-time CPU Monitoring",
                "Memory Usage Tracking",
                "Disk Usage Monitoring",
                "Network I/O Statistics",
                "Process Monitoring",
                "System Information"
            ],
            "endpoints": 4,
            "metrics": {
                "cpu_percent": round(cpu_percent, 1),
                "memory_percent": round(memory_percent, 1),
                "disk_percent": round(disk_percent, 1)
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def get_status_level(percent: float) -> str:
    """Get status level based on percentage"""
    if percent >= 90:
        return "critical"
    elif percent >= 75:
        return "warning"
    elif percent >= 50:
        return "moderate"
    else:
        return "good"

