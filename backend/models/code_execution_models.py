"""
Code Execution Models

Data models for secure code execution sandbox.
Supports Python and JavaScript execution in isolated Docker containers.

Author: GRETA PAI Development Team
Date: 2025-11-09
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, Literal, List
from datetime import datetime
from enum import Enum


class ExecutionLanguage(str, Enum):
    """Supported execution languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"


class ExecutionStatus(str, Enum):
    """Execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ResourceLimits(BaseModel):
    """Resource limits for code execution"""
    cpu_quota: int = Field(
        default=50000,
        ge=10000,
        le=100000,
        description="CPU quota in microseconds per 100ms period (10-100% of one core)"
    )
    cpu_period: int = Field(
        default=100000,
        description="CPU period in microseconds (default: 100ms)"
    )
    mem_limit: str = Field(
        default="256m",
        description="Memory limit (e.g., '256m', '512m', '1g')"
    )
    memswap_limit: str = Field(
        default="256m",
        description="Memory + swap limit (should equal mem_limit to disable swap)"
    )
    pids_limit: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Maximum number of processes"
    )
    storage_limit: str = Field(
        default="100m",
        description="Temporary storage limit"
    )

    @validator('mem_limit', 'memswap_limit', 'storage_limit')
    def validate_memory_format(cls, v):
        """Validate memory format (e.g., '256m', '1g')"""
        if not v:
            return v
        
        # Check if ends with valid unit
        valid_units = ['k', 'm', 'g']
        if v[-1].lower() not in valid_units:
            raise ValueError(f"Invalid memory format: {v}. Must end with k, m, or g")
        
        # Check if numeric part is valid
        try:
            int(v[:-1])
        except ValueError:
            raise ValueError(f"Invalid memory format: {v}. Numeric part must be an integer")
        
        return v.lower()


class CodeExecutionRequest(BaseModel):
    """Request to execute code in sandbox"""
    code: str = Field(
        ...,
        min_length=1,
        max_length=102400,  # 100 KB
        description="Code to execute (max 100 KB)"
    )
    language: ExecutionLanguage = Field(
        ...,
        description="Programming language"
    )
    timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Execution timeout in seconds (1-300)"
    )
    resource_limits: Optional[ResourceLimits] = Field(
        default=None,
        description="Custom resource limits (uses defaults if not specified)"
    )
    allow_network: bool = Field(
        default=False,
        description="Allow network access (disabled by default for security)"
    )
    environment: Optional[Dict[str, str]] = Field(
        default=None,
        description="Environment variables for execution"
    )
    agent_id: Optional[str] = Field(
        default=None,
        description="Agent ID for tracking and auditing"
    )
    stdin: Optional[str] = Field(
        default=None,
        description="Standard input for the code"
    )

    @validator('code')
    def validate_code_not_empty(cls, v):
        """Ensure code is not just whitespace"""
        if not v.strip():
            raise ValueError("Code cannot be empty or whitespace only")
        return v

    @validator('environment')
    def validate_environment(cls, v):
        """Validate environment variables"""
        if v is None:
            return v
        
        # Limit number of environment variables
        if len(v) > 50:
            raise ValueError("Maximum 50 environment variables allowed")
        
        # Validate keys and values
        for key, value in v.items():
            if not key or not isinstance(key, str):
                raise ValueError(f"Invalid environment variable key: {key}")
            if not isinstance(value, str):
                raise ValueError(f"Invalid environment variable value for {key}")
            if len(key) > 100 or len(value) > 1000:
                raise ValueError(f"Environment variable {key} too long")
        
        return v


class ResourceUsage(BaseModel):
    """Resource usage statistics"""
    cpu_percent: float = Field(
        default=0.0,
        description="CPU usage percentage"
    )
    memory_mb: float = Field(
        default=0.0,
        description="Memory usage in MB"
    )
    peak_memory_mb: float = Field(
        default=0.0,
        description="Peak memory usage in MB"
    )
    network_rx_bytes: int = Field(
        default=0,
        description="Network bytes received"
    )
    network_tx_bytes: int = Field(
        default=0,
        description="Network bytes transmitted"
    )


class CodeExecutionResult(BaseModel):
    """Result of code execution"""
    execution_id: str = Field(
        ...,
        description="Unique execution ID"
    )
    status: ExecutionStatus = Field(
        ...,
        description="Execution status"
    )
    success: bool = Field(
        ...,
        description="Whether execution completed successfully"
    )
    stdout: str = Field(
        default="",
        description="Standard output"
    )
    stderr: str = Field(
        default="",
        description="Standard error"
    )
    exit_code: int = Field(
        default=0,
        description="Process exit code"
    )
    execution_time: float = Field(
        default=0.0,
        ge=0,
        description="Execution time in seconds"
    )
    resource_usage: ResourceUsage = Field(
        default_factory=ResourceUsage,
        description="Resource consumption statistics"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if execution failed"
    )
    error_type: Optional[str] = Field(
        default=None,
        description="Error type (e.g., 'TimeoutError', 'SyntaxError')"
    )
    language: ExecutionLanguage = Field(
        ...,
        description="Programming language used"
    )
    agent_id: Optional[str] = Field(
        default=None,
        description="Agent ID that requested execution"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Execution timestamp"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Completion timestamp"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ValidationResult(BaseModel):
    """Result of code validation"""
    valid: bool = Field(
        ...,
        description="Whether code is valid"
    )
    language: ExecutionLanguage = Field(
        ...,
        description="Programming language"
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Validation errors"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Validation warnings"
    )
    forbidden_patterns: List[str] = Field(
        default_factory=list,
        description="Detected forbidden patterns"
    )
    code_size: int = Field(
        default=0,
        description="Code size in bytes"
    )


class ContainerInfo(BaseModel):
    """Information about execution container"""
    container_id: str = Field(
        ...,
        description="Docker container ID"
    )
    image: str = Field(
        ...,
        description="Docker image used"
    )
    status: str = Field(
        ...,
        description="Container status"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Container creation time"
    )
    removed_at: Optional[datetime] = Field(
        default=None,
        description="Container removal time"
    )


class ExecutionStatistics(BaseModel):
    """Execution statistics for monitoring"""
    total_executions: int = Field(
        default=0,
        description="Total number of executions"
    )
    successful_executions: int = Field(
        default=0,
        description="Number of successful executions"
    )
    failed_executions: int = Field(
        default=0,
        description="Number of failed executions"
    )
    timeout_executions: int = Field(
        default=0,
        description="Number of timeout executions"
    )
    average_execution_time: float = Field(
        default=0.0,
        description="Average execution time in seconds"
    )
    average_memory_usage: float = Field(
        default=0.0,
        description="Average memory usage in MB"
    )
    executions_by_language: Dict[str, int] = Field(
        default_factory=dict,
        description="Execution count by language"
    )
    executions_by_agent: Dict[str, int] = Field(
        default_factory=dict,
        description="Execution count by agent"
    )
    last_updated: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last statistics update"
    )

