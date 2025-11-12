"""
Code Execution Service

Secure code execution sandbox using Docker containers.
Supports Python and JavaScript with comprehensive security controls.

Author: GRETA PAI Development Team
Date: 2025-11-09
"""

import asyncio
import docker
import uuid
import time
import ast
import re
import hashlib
from typing import Optional, Dict, Any, List
from datetime import datetime
from loguru import logger

from models.code_execution_models import (
    CodeExecutionRequest,
    CodeExecutionResult,
    ValidationResult,
    ResourceLimits,
    ResourceUsage,
    ExecutionStatus,
    ExecutionLanguage,
    ExecutionStatistics,
    ContainerInfo
)
from database import Database


class CodeExecutionService:
    """
    Secure code execution sandbox service
    
    Features:
    - Docker-based isolation
    - Resource limits (CPU, memory, disk)
    - Execution timeouts
    - Network isolation
    - Code validation
    - Comprehensive logging
    """
    
    def __init__(self, db: Database):
        """Initialize code execution service"""
        self.db = db
        self.executions_collection = "code_executions"
        self.statistics_collection = "code_execution_stats"
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
            logger.info("✅ Docker client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Docker client: {e}")
            raise
        
        # Default resource limits
        self.default_limits = ResourceLimits()
        
        # Docker images for each language - using custom sandbox images
        self.images = {
            ExecutionLanguage.PYTHON: "greta-pai/python-sandbox:latest",
            ExecutionLanguage.JAVASCRIPT: "greta-pai/javascript-sandbox:latest"
        }
        
        # Forbidden patterns (configurable)
        self.forbidden_patterns = {
            ExecutionLanguage.PYTHON: [
                r"import\s+os",
                r"import\s+subprocess",
                r"from\s+os\s+import",
                r"from\s+subprocess\s+import",
                r"__import__\s*\(",
                r"eval\s*\(",
                r"exec\s*\(",
                r"compile\s*\(",
            ],
            ExecutionLanguage.JAVASCRIPT: [
                r"require\s*\(\s*['\"]child_process['\"]",
                r"require\s*\(\s*['\"]fs['\"]",
                r"eval\s*\(",
                r"Function\s*\(",
                r"require\s*\(\s*['\"]net['\"]",
                r"require\s*\(\s*['\"]http['\"]",
            ]
        }
        
        # Statistics
        self.stats = ExecutionStatistics()
        
        logger.info("✅ Code Execution Service initialized")
    
    async def execute_code(
        self,
        request: CodeExecutionRequest
    ) -> CodeExecutionResult:
        """
        Execute code in isolated Docker container
        
        Args:
            request: Code execution request
            
        Returns:
            Execution result with output and statistics
        """
        execution_id = f"exec_{uuid.uuid4().hex[:12]}"
        start_time = time.time()
        
        logger.info(f"🚀 Starting code execution {execution_id}")
        logger.info(f"   Language: {request.language}")
        logger.info(f"   Code size: {len(request.code)} bytes")
        logger.info(f"   Timeout: {request.timeout}s")
        
        try:
            # Step 1: Validate code
            validation = await self.validate_code(request.code, request.language)
            if not validation.valid:
                logger.error(f"❌ Code validation failed: {validation.errors}")
                return self._create_error_result(
                    execution_id=execution_id,
                    language=request.language,
                    agent_id=request.agent_id,
                    error="Code validation failed",
                    error_type="ValidationError",
                    details="; ".join(validation.errors),
                    execution_time=time.time() - start_time
                )
            
            # Step 2: Prepare container configuration
            container_config = self._prepare_container_config(request)
            
            # Step 3: Execute code in container
            result = await self._execute_in_container(
                execution_id=execution_id,
                code=request.code,
                language=request.language,
                config=container_config,
                timeout=request.timeout,
                stdin=request.stdin
            )
            
            # Step 4: Calculate execution time
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            result.agent_id = request.agent_id
            result.completed_at = datetime.utcnow()
            
            # Step 5: Store result
            await self._store_execution(result)
            
            # Step 6: Update statistics
            await self._update_statistics(result)
            
            logger.info(f"✅ Execution {execution_id} completed in {execution_time:.2f}s")
            logger.info(f"   Status: {result.status}")
            logger.info(f"   Exit code: {result.exit_code}")
            
            return result
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            logger.error(f"⏱️ Execution {execution_id} timed out after {execution_time:.2f}s")
            
            result = self._create_error_result(
                execution_id=execution_id,
                language=request.language,
                agent_id=request.agent_id,
                error=f"Execution timeout after {request.timeout} seconds",
                error_type="TimeoutError",
                execution_time=execution_time
            )
            result.status = ExecutionStatus.TIMEOUT
            
            await self._store_execution(result)
            await self._update_statistics(result)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Execution {execution_id} failed: {e}")
            
            result = self._create_error_result(
                execution_id=execution_id,
                language=request.language,
                agent_id=request.agent_id,
                error=str(e),
                error_type=type(e).__name__,
                execution_time=execution_time
            )
            
            await self._store_execution(result)
            await self._update_statistics(result)
            
            return result
    
    async def validate_code(
        self,
        code: str,
        language: ExecutionLanguage
    ) -> ValidationResult:
        """
        Validate code before execution
        
        Args:
            code: Code to validate
            language: Programming language
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        forbidden = []
        
        # Check code size
        code_size = len(code.encode('utf-8'))
        if code_size > 102400:  # 100 KB
            errors.append(f"Code size ({code_size} bytes) exceeds maximum (102400 bytes)")
        
        # Check for forbidden patterns
        patterns = self.forbidden_patterns.get(language, [])
        for pattern in patterns:
            if re.search(pattern, code, re.IGNORECASE):
                forbidden.append(pattern)
                errors.append(f"Forbidden pattern detected: {pattern}")
                warnings.append(f"Security violation: Code contains dangerous operation")
        
        # Language-specific validation
        if language == ExecutionLanguage.PYTHON:
            # Check Python syntax
            try:
                ast.parse(code)
            except SyntaxError as e:
                errors.append(f"Python syntax error: {e}")
        
        elif language == ExecutionLanguage.JAVASCRIPT:
            # Basic JavaScript validation (check for common syntax errors)
            if code.count('{') != code.count('}'):
                errors.append("Mismatched curly braces")
            if code.count('(') != code.count(')'):
                errors.append("Mismatched parentheses")
            if code.count('[') != code.count(']'):
                errors.append("Mismatched square brackets")
        
        valid = len(errors) == 0
        
        return ValidationResult(
            valid=valid,
            language=language,
            errors=errors,
            warnings=warnings,
            forbidden_patterns=forbidden,
            code_size=code_size
        )
    
    def _prepare_container_config(
        self,
        request: CodeExecutionRequest
    ) -> Dict[str, Any]:
        """
        Prepare Docker container configuration
        
        Args:
            request: Code execution request
            
        Returns:
            Container configuration dictionary
        """
        # Use custom limits or defaults
        limits = request.resource_limits or self.default_limits
        
        # Base configuration
        config = {
            "image": self.images[request.language],
            "network_mode": "none" if not request.allow_network else "bridge",
            "read_only": True,  # Read-only root filesystem
            "user": "nobody",  # Non-root user
            "cap_drop": ["ALL"],  # Drop all capabilities
            "security_opt": ["no-new-privileges"],
            "tmpfs": {"/tmp": f"size={limits.storage_limit},mode=1777,noexec"},  # Add noexec flag
            "mem_limit": limits.mem_limit,
            "memswap_limit": limits.memswap_limit,
            "cpu_quota": limits.cpu_quota,
            "cpu_period": limits.cpu_period,
            "pids_limit": limits.pids_limit,
        }
        
        # Add environment variables if provided
        if request.environment:
            config["environment"] = request.environment
        
        return config
    
    async def _execute_in_container(
        self,
        execution_id: str,
        code: str,
        language: ExecutionLanguage,
        config: Dict[str, Any],
        timeout: int,
        stdin: Optional[str] = None
    ) -> CodeExecutionResult:
        """
        Execute code in Docker container
        
        This method runs in a thread pool to avoid blocking the event loop.
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._execute_in_container_sync,
            execution_id,
            code,
            language,
            config,
            timeout,
            stdin
        )
    
    def _execute_in_container_sync(
        self,
        execution_id: str,
        code: str,
        language: ExecutionLanguage,
        config: Dict[str, Any],
        timeout: int,
        stdin: Optional[str] = None
    ) -> CodeExecutionResult:
        """
        Synchronous container execution (runs in thread pool)
        """
        container = None
        start_time = time.time()

        try:
            # Prepare command based on language
            if language == ExecutionLanguage.PYTHON:
                command = ["python", "-c", code]
            elif language == ExecutionLanguage.JAVASCRIPT:
                command = ["node", "-e", code]
            else:
                raise ValueError(f"Unsupported language: {language}")

            # Add command to config
            config["command"] = command

            # Add stdin if provided
            if stdin:
                config["stdin_open"] = True

            # IMPORTANT: Set detach=True to get container object (not output)
            config["detach"] = True

            logger.debug(f"🐳 Creating container for {execution_id}")

            # Run container (detached)
            container = self.docker_client.containers.run(**config)

            # Wait for container with timeout
            try:
                result = container.wait(timeout=timeout)
                exit_code = result.get('StatusCode', 0)

                # Get logs
                stdout = container.logs(stdout=True, stderr=False).decode('utf-8')
                stderr = container.logs(stdout=False, stderr=True).decode('utf-8')

                execution_time = time.time() - start_time

                logger.debug(f"✅ Container execution completed in {execution_time:.2f}s")

                # Remove container
                try:
                    container.remove(force=True)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to remove container: {e}")

                # Determine status based on exit code
                status = ExecutionStatus.COMPLETED if exit_code == 0 else ExecutionStatus.FAILED

                return CodeExecutionResult(
                    execution_id=execution_id,
                    status=status,
                    success=exit_code == 0,
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=exit_code,
                    execution_time=execution_time,
                    language=language,
                    resource_usage=ResourceUsage(),  # TODO: Extract from container stats
                    created_at=datetime.utcnow()
                )

            except Exception as timeout_error:
                # Timeout occurred - kill the container
                execution_time = time.time() - start_time
                logger.error(f"⏱️ Container {execution_id} timed out after {execution_time:.2f}s - killing container")

                try:
                    container.kill()
                    container.remove(force=True)
                except Exception as e:
                    logger.error(f"❌ Failed to kill/remove container: {e}")

                return CodeExecutionResult(
                    execution_id=execution_id,
                    status=ExecutionStatus.TIMEOUT,
                    success=False,
                    stdout="",
                    stderr=f"Execution timed out after {timeout} seconds",
                    exit_code=-1,
                    execution_time=execution_time,
                    language=language,
                    resource_usage=ResourceUsage(),
                    created_at=datetime.utcnow(),
                    error="Execution timeout",
                    error_type="TimeoutError"
                )
            
        except docker.errors.ContainerError as e:
            # Container exited with non-zero code
            execution_time = time.time() - start_time

            # Clean up container if it exists
            if container:
                try:
                    container.remove(force=True)
                except Exception as cleanup_error:
                    logger.warning(f"⚠️ Failed to remove container: {cleanup_error}")

            return CodeExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.FAILED,
                success=False,
                stdout=e.stdout.decode('utf-8') if e.stdout else "",
                stderr=e.stderr.decode('utf-8') if e.stderr else str(e),
                exit_code=e.exit_status,
                execution_time=execution_time,
                language=language,
                error=f"Container exited with code {e.exit_status}",
                error_type="ContainerError",
                resource_usage=ResourceUsage(),
                created_at=datetime.utcnow()
            )

        except Exception as e:
            execution_time = time.time() - start_time

            # Clean up container if it exists
            if container:
                try:
                    container.kill()
                    container.remove(force=True)
                except Exception as cleanup_error:
                    logger.warning(f"⚠️ Failed to kill/remove container: {cleanup_error}")

            return CodeExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.FAILED,
                success=False,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                execution_time=execution_time,
                language=language,
                error=str(e),
                error_type=type(e).__name__,
                resource_usage=ResourceUsage(),
                created_at=datetime.utcnow()
            )

    def _create_error_result(
        self,
        execution_id: str,
        language: ExecutionLanguage,
        agent_id: Optional[str],
        error: str,
        error_type: str,
        execution_time: float = 0.0,
        details: str = ""
    ) -> CodeExecutionResult:
        """Create error result"""
        stderr = error
        if details:
            stderr += f"\nDetails: {details}"

        return CodeExecutionResult(
            execution_id=execution_id,
            status=ExecutionStatus.FAILED,
            success=False,
            stdout="",
            stderr=stderr,
            exit_code=-1,
            execution_time=execution_time,
            language=language,
            agent_id=agent_id,
            error=error,
            error_type=error_type,
            resource_usage=ResourceUsage(),
            created_at=datetime.utcnow()
        )

    async def _store_execution(self, result: CodeExecutionResult):
        """Store execution result in database"""
        try:
            execution_data = result.model_dump()
            await self.db.insert_one(self.executions_collection, execution_data)
            logger.debug(f"💾 Stored execution {result.execution_id}")
        except Exception as e:
            logger.error(f"❌ Failed to store execution: {e}")

    async def _update_statistics(self, result: CodeExecutionResult):
        """Update execution statistics"""
        try:
            self.stats.total_executions += 1

            if result.success:
                self.stats.successful_executions += 1
            elif result.status == ExecutionStatus.TIMEOUT:
                self.stats.timeout_executions += 1
            else:
                self.stats.failed_executions += 1

            # Update averages
            total = self.stats.total_executions
            self.stats.average_execution_time = (
                (self.stats.average_execution_time * (total - 1) + result.execution_time) / total
            )

            # Update by language
            lang_key = result.language.value
            self.stats.executions_by_language[lang_key] = (
                self.stats.executions_by_language.get(lang_key, 0) + 1
            )

            # Update by agent
            if result.agent_id:
                self.stats.executions_by_agent[result.agent_id] = (
                    self.stats.executions_by_agent.get(result.agent_id, 0) + 1
                )

            self.stats.last_updated = datetime.utcnow()

            # Store statistics in database (async)
            asyncio.create_task(self._store_statistics())

        except Exception as e:
            logger.error(f"❌ Failed to update statistics: {e}")

    async def _store_statistics(self):
        """Store statistics in database"""
        try:
            stats_data = self.stats.model_dump()
            stats_data["_id"] = "global"

            # Try to find existing statistics
            existing = await self.db.find_one(
                self.statistics_collection,
                {"_id": "global"}
            )

            if existing:
                # Update existing
                await self.db.update_one(
                    self.statistics_collection,
                    {"_id": "global"},
                    {"$set": stats_data}
                )
            else:
                # Insert new
                await self.db.insert_one(
                    self.statistics_collection,
                    stats_data
                )
        except Exception as e:
            logger.error(f"❌ Failed to store statistics: {e}")

    async def get_execution(
        self,
        execution_id: str
    ) -> Optional[CodeExecutionResult]:
        """
        Retrieve execution result by ID

        Args:
            execution_id: Execution ID

        Returns:
            Execution result or None if not found
        """
        try:
            data = await self.db.find_one(
                self.executions_collection,
                {"execution_id": execution_id}
            )

            if not data:
                return None

            # Convert datetime strings to datetime objects
            for key in ["created_at", "completed_at"]:
                if key in data and data[key] and isinstance(data[key], str):
                    data[key] = datetime.fromisoformat(data[key])

            return CodeExecutionResult(**data)

        except Exception as e:
            logger.error(f"❌ Failed to get execution: {e}")
            return None

    async def list_executions(
        self,
        agent_id: Optional[str] = None,
        language: Optional[ExecutionLanguage] = None,
        status: Optional[ExecutionStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[CodeExecutionResult]:
        """
        List recent executions

        Args:
            agent_id: Filter by agent ID
            language: Filter by language
            status: Filter by status
            limit: Maximum results
            offset: Results offset

        Returns:
            List of execution results
        """
        try:
            # Build query
            query = {}
            if agent_id:
                query["agent_id"] = agent_id
            if language:
                query["language"] = language.value
            if status:
                query["status"] = status.value

            # Query database
            cursor = self.db.db[self.executions_collection].find(query).sort(
                "created_at", -1
            ).skip(offset).limit(limit)

            results = []
            async for data in cursor:
                # Convert datetime strings
                for key in ["created_at", "completed_at"]:
                    if key in data and data[key] and isinstance(data[key], str):
                        data[key] = datetime.fromisoformat(data[key])

                results.append(CodeExecutionResult(**data))

            return results

        except Exception as e:
            logger.error(f"❌ Failed to list executions: {e}")
            return []

    async def get_statistics(self) -> ExecutionStatistics:
        """
        Get execution statistics

        Returns:
            Execution statistics
        """
        return self.stats

    async def cleanup_containers(self) -> int:
        """
        Cleanup stale containers (maintenance)

        Returns:
            Number of containers removed
        """
        try:
            # List all containers with our label
            containers = self.docker_client.containers.list(
                all=True,
                filters={"label": "greta-pai-sandbox"}
            )

            removed = 0
            for container in containers:
                try:
                    container.remove(force=True)
                    removed += 1
                    logger.debug(f"🗑️ Removed container {container.id[:12]}")
                except Exception as e:
                    logger.error(f"❌ Failed to remove container {container.id[:12]}: {e}")

            logger.info(f"✅ Cleaned up {removed} containers")
            return removed

        except Exception as e:
            logger.error(f"❌ Failed to cleanup containers: {e}")
            return 0

    async def health_check(self) -> Dict[str, Any]:
        """
        Health check for code execution service

        Returns:
            Health status
        """
        try:
            # Check Docker connection
            self.docker_client.ping()
            docker_healthy = True
        except Exception as e:
            logger.error(f"❌ Docker health check failed: {e}")
            docker_healthy = False

        # Check images availability
        images_available = {}
        for lang, image in self.images.items():
            try:
                self.docker_client.images.get(image)
                images_available[lang.value] = True
            except docker.errors.ImageNotFound:
                images_available[lang.value] = False
                logger.warning(f"⚠️ Image not found: {image}")

        return {
            "healthy": docker_healthy and all(images_available.values()),
            "docker_connected": docker_healthy,
            "images_available": images_available,
            "total_executions": self.stats.total_executions,
            "success_rate": (
                self.stats.successful_executions / self.stats.total_executions * 100
                if self.stats.total_executions > 0 else 0
            )
        }

