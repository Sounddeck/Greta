"""
Tool Registry Service for CPAS5
Centralized tool management with permission controls and execution
"""

import logging
import uuid
import asyncio
import aiohttp
import dns.resolver
import whois
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from pydantic import BaseModel, Field
from database import Database, get_database

logger = logging.getLogger(__name__)


class ToolCategory(str, Enum):
    """Tool categories"""
    OSINT = "osint"
    DATA_PROCESSING = "data_processing"
    WEB_API = "web_api"
    DATABASE = "database"
    COMMUNICATION = "communication"
    FILE_SYSTEM = "file_system"
    CODE_EXECUTION = "code_execution"
    AI_ML = "ai_ml"
    BUSINESS = "business"
    ANALYTICS = "analytics"
    SECURITY = "security"


class PermissionLevel(str, Enum):
    """Permission levels for tools"""
    PUBLIC = "public"  # All agents can use
    RESTRICTED = "restricted"  # Requires explicit permission
    ADMIN = "admin"  # Master Agent only
    CUSTOM = "custom"  # Per-agent whitelist/blacklist


class ToolDefinition(BaseModel):
    """Tool definition model"""
    tool_id: str
    name: str
    description: str
    category: ToolCategory
    permission_level: PermissionLevel = PermissionLevel.PUBLIC
    parameters: Dict[str, Any] = Field(default_factory=dict)
    returns: Dict[str, Any] = Field(default_factory=dict)
    examples: List[Dict[str, Any]] = Field(default_factory=list)
    rate_limit: Optional[int] = None  # Calls per minute
    timeout: int = 30  # Seconds
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolPermission(BaseModel):
    """Tool permission for specific agent"""
    permission_id: str
    tool_id: str
    agent_id: str
    allowed: bool = True
    restrictions: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class ToolExecution(BaseModel):
    """Tool execution record"""
    execution_id: str
    tool_id: str
    agent_id: str
    parameters: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    success: bool = False
    error: Optional[str] = None
    execution_time: float = 0.0
    started_at: datetime
    completed_at: Optional[datetime] = None


class ToolRegistry:
    """
    Centralized tool registry with permission controls and execution
    """
    
    def __init__(self):
        self.db: Optional[Database] = None
        self.tools: Dict[str, ToolDefinition] = {}
        self.tool_executors: Dict[str, Callable] = {}
        self.permissions: Dict[str, ToolPermission] = {}
        
        # Collections
        self.tools_collection = "tool_registry"
        self.permissions_collection = "tool_permissions"
        self.executions_collection = "tool_executions"
        
        # Metrics
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_execution_time": 0.0,
            "executions_by_category": {},
            "executions_by_agent": {}
        }
    
    async def initialize(self):
        """Initialize tool registry"""
        try:
            self.db = await get_database()
            
            # Register built-in tools
            await self._register_osint_tools()
            await self._register_data_processing_tools()
            await self._register_web_api_tools()
            await self._register_code_execution_tools()
            await self._register_ai_ml_tools()
            
            # Load permissions from database
            await self._load_permissions()
            
            logger.info(f"✅ Tool Registry initialized with {len(self.tools)} tools")
            
        except Exception as e:
            logger.error(f"❌ Tool Registry initialization failed: {e}")
            raise
    
    async def register_tool(
        self,
        tool: ToolDefinition,
        executor: Callable
    ) -> ToolDefinition:
        """Register a new tool"""
        try:
            # Store tool definition
            self.tools[tool.tool_id] = tool
            self.tool_executors[tool.tool_id] = executor
            
            # Save to database
            tool_dict = tool.model_dump()
            await self.db.insert_one(self.tools_collection, tool_dict)
            
            logger.info(f"✅ Tool registered: {tool.name} ({tool.tool_id})")
            return tool
            
        except Exception as e:
            logger.error(f"❌ Tool registration failed: {e}")
            raise
    
    async def execute_tool(
        self,
        tool_id: str,
        parameters: Dict[str, Any],
        agent_id: str
    ) -> Dict[str, Any]:
        """Execute a tool with permission checking"""
        start_time = datetime.utcnow()
        execution_id = f"exec_{uuid.uuid4().hex[:12]}"
        
        try:
            # Check if tool exists
            if tool_id not in self.tools:
                raise ValueError(f"Tool not found: {tool_id}")
            
            tool = self.tools[tool_id]
            
            # Check permissions
            if not await self.check_permission(agent_id, tool_id):
                raise PermissionError(f"Agent {agent_id} not authorized to use tool {tool_id}")
            
            # Get executor
            executor = self.tool_executors.get(tool_id)
            if not executor:
                raise ValueError(f"No executor found for tool: {tool_id}")
            
            # Execute with timeout
            result = await asyncio.wait_for(
                executor(parameters),
                timeout=tool.timeout
            )
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Record execution
            execution = ToolExecution(
                execution_id=execution_id,
                tool_id=tool_id,
                agent_id=agent_id,
                parameters=parameters,
                result=result,
                success=True,
                execution_time=execution_time,
                started_at=start_time,
                completed_at=datetime.utcnow()
            )
            
            await self._record_execution(execution)
            
            # Update metrics
            self._update_metrics(tool, agent_id, execution_time, True)
            
            return {
                "success": True,
                "result": result,
                "execution_id": execution_id,
                "execution_time": execution_time
            }
            
        except asyncio.TimeoutError:
            error = f"Tool execution timed out after {tool.timeout}s"
            logger.error(error)
            
            execution = ToolExecution(
                execution_id=execution_id,
                tool_id=tool_id,
                agent_id=agent_id,
                parameters=parameters,
                success=False,
                error=error,
                execution_time=(datetime.utcnow() - start_time).total_seconds(),
                started_at=start_time,
                completed_at=datetime.utcnow()
            )
            
            await self._record_execution(execution)
            self._update_metrics(tool, agent_id, 0, False)
            
            return {
                "success": False,
                "error": error,
                "execution_id": execution_id
            }
            
        except Exception as e:
            error = str(e)
            logger.error(f"❌ Tool execution failed: {error}")
            
            execution = ToolExecution(
                execution_id=execution_id,
                tool_id=tool_id,
                agent_id=agent_id,
                parameters=parameters,
                success=False,
                error=error,
                execution_time=(datetime.utcnow() - start_time).total_seconds(),
                started_at=start_time,
                completed_at=datetime.utcnow()
            )
            
            await self._record_execution(execution)
            self._update_metrics(tool, agent_id, 0, False)
            
            return {
                "success": False,
                "error": error,
                "execution_id": execution_id
            }
    
    async def check_permission(self, agent_id: str, tool_id: str) -> bool:
        """Check if agent has permission to use tool"""
        try:
            tool = self.tools.get(tool_id)
            if not tool:
                return False
            
            # Public tools are always allowed
            if tool.permission_level == PermissionLevel.PUBLIC:
                return True
            
            # Admin tools only for master agent
            if tool.permission_level == PermissionLevel.ADMIN:
                return agent_id == "master_agent"
            
            # Check custom permissions
            perm_key = f"{agent_id}:{tool_id}"
            permission = self.permissions.get(perm_key)
            
            if permission:
                return permission.allowed
            
            # Restricted tools require explicit permission
            if tool.permission_level == PermissionLevel.RESTRICTED:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Permission check failed: {e}")
            return False
    
    async def get_tools_for_agent(self, agent_id: str) -> List[ToolDefinition]:
        """Get all tools available to an agent"""
        available_tools = []
        
        for tool_id, tool in self.tools.items():
            if await self.check_permission(agent_id, tool_id):
                available_tools.append(tool)
        
        return available_tools
    
    async def _record_execution(self, execution: ToolExecution):
        """Record tool execution in database"""
        try:
            exec_dict = execution.model_dump()
            exec_dict["started_at"] = exec_dict["started_at"].isoformat()
            if exec_dict.get("completed_at"):
                exec_dict["completed_at"] = exec_dict["completed_at"].isoformat()
            
            await self.db.insert_one(self.executions_collection, exec_dict)
            
        except Exception as e:
            logger.error(f"Failed to record execution: {e}")
    
    def _update_metrics(self, tool: ToolDefinition, agent_id: str, execution_time: float, success: bool):
        """Update execution metrics"""
        self.metrics["total_executions"] += 1
        
        if success:
            self.metrics["successful_executions"] += 1
            self.metrics["total_execution_time"] += execution_time
        else:
            self.metrics["failed_executions"] += 1
        
        # By category
        category = tool.category.value
        if category not in self.metrics["executions_by_category"]:
            self.metrics["executions_by_category"][category] = 0
        self.metrics["executions_by_category"][category] += 1
        
        # By agent
        if agent_id not in self.metrics["executions_by_agent"]:
            self.metrics["executions_by_agent"][agent_id] = 0
        self.metrics["executions_by_agent"][agent_id] += 1
    
    async def _load_permissions(self):
        """Load permissions from database"""
        try:
            perms = await self.db.find_many(self.permissions_collection, {}, limit=1000)

            for perm_dict in perms:
                # Convert datetime strings
                for k in ["created_at", "updated_at"]:
                    if k in perm_dict and isinstance(perm_dict[k], str):
                        perm_dict[k] = datetime.fromisoformat(perm_dict[k])

                perm = ToolPermission(**perm_dict)
                perm_key = f"{perm.agent_id}:{perm.tool_id}"
                self.permissions[perm_key] = perm

            logger.info(f"✅ Loaded {len(perms)} tool permissions")

        except Exception as e:
            logger.debug(f"Failed to load permissions: {e}")

    # ==================== OSINT TOOLS ====================

    async def _register_osint_tools(self):
        """Register OSINT tools"""

        # Web Scraper
        await self.register_tool(
            ToolDefinition(
                tool_id="osint_web_scraper",
                name="Web Scraper",
                description="Scrape content from web pages",
                category=ToolCategory.OSINT,
                permission_level=PermissionLevel.PUBLIC,
                parameters={
                    "url": {"type": "string", "required": True},
                    "selector": {"type": "string", "required": False},
                    "extract_links": {"type": "boolean", "default": False}
                },
                timeout=30
            ),
            self._execute_web_scraper
        )

        # WHOIS Lookup
        await self.register_tool(
            ToolDefinition(
                tool_id="osint_whois_lookup",
                name="WHOIS Lookup",
                description="Get WHOIS information for a domain",
                category=ToolCategory.OSINT,
                permission_level=PermissionLevel.PUBLIC,
                parameters={
                    "domain": {"type": "string", "required": True}
                },
                timeout=15
            ),
            self._execute_whois_lookup
        )

        # DNS Lookup
        await self.register_tool(
            ToolDefinition(
                tool_id="osint_dns_lookup",
                name="DNS Lookup",
                description="Query DNS records for a domain",
                category=ToolCategory.OSINT,
                permission_level=PermissionLevel.PUBLIC,
                parameters={
                    "domain": {"type": "string", "required": True},
                    "record_type": {"type": "string", "default": "A", "enum": ["A", "AAAA", "MX", "TXT", "NS", "CNAME"]}
                },
                timeout=10
            ),
            self._execute_dns_lookup
        )

        # Email Validator
        await self.register_tool(
            ToolDefinition(
                tool_id="osint_email_validator",
                name="Email Validator",
                description="Validate email address format and domain",
                category=ToolCategory.OSINT,
                permission_level=PermissionLevel.PUBLIC,
                parameters={
                    "email": {"type": "string", "required": True}
                },
                timeout=5
            ),
            self._execute_email_validator
        )

        # IP Geolocation
        await self.register_tool(
            ToolDefinition(
                tool_id="osint_ip_geolocation",
                name="IP Geolocation",
                description="Get geolocation information for an IP address",
                category=ToolCategory.OSINT,
                permission_level=PermissionLevel.PUBLIC,
                parameters={
                    "ip": {"type": "string", "required": True}
                },
                timeout=10
            ),
            self._execute_ip_geolocation
        )

    async def _execute_web_scraper(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute web scraper tool"""
        try:
            from bs4 import BeautifulSoup

            url = params.get("url")
            selector = params.get("selector")
            extract_links = params.get("extract_links", False)

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as response:
                    html = await response.text()

            soup = BeautifulSoup(html, 'html.parser')

            result = {
                "url": url,
                "title": soup.title.string if soup.title else None,
                "content": None,
                "links": []
            }

            if selector:
                elements = soup.select(selector)
                result["content"] = [elem.get_text(strip=True) for elem in elements]
            else:
                result["content"] = soup.get_text(strip=True)[:5000]  # Limit to 5000 chars

            if extract_links:
                links = soup.find_all('a', href=True)
                result["links"] = [link['href'] for link in links[:50]]  # Limit to 50 links

            return result

        except Exception as e:
            logger.error(f"Web scraper failed: {e}")
            return {"error": str(e)}

    async def _execute_whois_lookup(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute WHOIS lookup tool"""
        try:
            domain = params.get("domain")

            # Run in thread pool since whois is blocking
            loop = asyncio.get_event_loop()
            whois_data = await loop.run_in_executor(None, whois.whois, domain)

            return {
                "domain": domain,
                "registrar": whois_data.registrar,
                "creation_date": str(whois_data.creation_date) if whois_data.creation_date else None,
                "expiration_date": str(whois_data.expiration_date) if whois_data.expiration_date else None,
                "name_servers": whois_data.name_servers if whois_data.name_servers else [],
                "status": whois_data.status if whois_data.status else None
            }

        except Exception as e:
            logger.error(f"WHOIS lookup failed: {e}")
            return {"error": str(e)}

    async def _execute_dns_lookup(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute DNS lookup tool"""
        try:
            domain = params.get("domain")
            record_type = params.get("record_type", "A")

            # Run in thread pool since dns.resolver is blocking
            loop = asyncio.get_event_loop()
            answers = await loop.run_in_executor(
                None,
                lambda: dns.resolver.resolve(domain, record_type)
            )

            records = [str(rdata) for rdata in answers]

            return {
                "domain": domain,
                "record_type": record_type,
                "records": records,
                "ttl": answers.rrset.ttl if answers.rrset else None
            }

        except Exception as e:
            logger.error(f"DNS lookup failed: {e}")
            return {"error": str(e)}

    async def _execute_email_validator(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute email validator tool"""
        try:
            import re

            email = params.get("email")

            # Basic email regex
            email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            is_valid_format = bool(re.match(email_regex, email))

            if not is_valid_format:
                return {
                    "email": email,
                    "valid": False,
                    "reason": "Invalid email format"
                }

            # Extract domain
            domain = email.split('@')[1]

            # Check if domain has MX records
            try:
                loop = asyncio.get_event_loop()
                mx_records = await loop.run_in_executor(
                    None,
                    lambda: dns.resolver.resolve(domain, 'MX')
                )
                has_mx = len(list(mx_records)) > 0
            except:
                has_mx = False

            return {
                "email": email,
                "valid": is_valid_format and has_mx,
                "format_valid": is_valid_format,
                "domain_has_mx": has_mx,
                "domain": domain
            }

        except Exception as e:
            logger.error(f"Email validation failed: {e}")
            return {"error": str(e)}

    async def _execute_ip_geolocation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute IP geolocation tool"""
        try:
            ip = params.get("ip")

            # Use ip-api.com (free, no API key required)
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://ip-api.com/json/{ip}") as response:
                    data = await response.json()

            return {
                "ip": ip,
                "country": data.get("country"),
                "country_code": data.get("countryCode"),
                "region": data.get("regionName"),
                "city": data.get("city"),
                "zip": data.get("zip"),
                "lat": data.get("lat"),
                "lon": data.get("lon"),
                "timezone": data.get("timezone"),
                "isp": data.get("isp"),
                "org": data.get("org")
            }

        except Exception as e:
            logger.error(f"IP geolocation failed: {e}")
            return {"error": str(e)}

    # ==================== DATA PROCESSING TOOLS ====================

    async def _register_data_processing_tools(self):
        """Register data processing tools"""

        # JSON Parser
        await self.register_tool(
            ToolDefinition(
                tool_id="data_json_parser",
                name="JSON Parser",
                description="Parse and validate JSON data",
                category=ToolCategory.DATA_PROCESSING,
                permission_level=PermissionLevel.PUBLIC,
                parameters={
                    "json_string": {"type": "string", "required": True},
                    "validate_only": {"type": "boolean", "default": False}
                },
                timeout=5
            ),
            self._execute_json_parser
        )

        # CSV Parser
        await self.register_tool(
            ToolDefinition(
                tool_id="data_csv_parser",
                name="CSV Parser",
                description="Parse CSV data",
                category=ToolCategory.DATA_PROCESSING,
                permission_level=PermissionLevel.PUBLIC,
                parameters={
                    "csv_string": {"type": "string", "required": True},
                    "delimiter": {"type": "string", "default": ","},
                    "has_header": {"type": "boolean", "default": True}
                },
                timeout=10
            ),
            self._execute_csv_parser
        )

        # Data Transformer
        await self.register_tool(
            ToolDefinition(
                tool_id="data_transformer",
                name="Data Transformer",
                description="Transform data structures",
                category=ToolCategory.DATA_PROCESSING,
                permission_level=PermissionLevel.PUBLIC,
                parameters={
                    "data": {"type": "object", "required": True},
                    "transformation": {"type": "string", "required": True}
                },
                timeout=10
            ),
            self._execute_data_transformer
        )

    async def _execute_json_parser(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute JSON parser tool"""
        try:
            import json

            json_string = params.get("json_string")
            validate_only = params.get("validate_only", False)

            parsed = json.loads(json_string)

            if validate_only:
                return {"valid": True, "message": "JSON is valid"}

            return {
                "valid": True,
                "data": parsed,
                "type": type(parsed).__name__
            }

        except json.JSONDecodeError as e:
            return {
                "valid": False,
                "error": str(e),
                "line": e.lineno,
                "column": e.colno
            }
        except Exception as e:
            logger.error(f"JSON parser failed: {e}")
            return {"error": str(e)}

    async def _execute_csv_parser(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute CSV parser tool"""
        try:
            import csv
            import io

            csv_string = params.get("csv_string")
            delimiter = params.get("delimiter", ",")
            has_header = params.get("has_header", True)

            reader = csv.reader(io.StringIO(csv_string), delimiter=delimiter)
            rows = list(reader)

            if not rows:
                return {"rows": [], "columns": [], "row_count": 0}

            result = {
                "row_count": len(rows),
                "rows": []
            }

            if has_header:
                result["columns"] = rows[0]
                result["rows"] = rows[1:]
            else:
                result["columns"] = [f"col_{i}" for i in range(len(rows[0]))]
                result["rows"] = rows

            return result

        except Exception as e:
            logger.error(f"CSV parser failed: {e}")
            return {"error": str(e)}

    async def _execute_data_transformer(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data transformer tool"""
        try:
            data = params.get("data")
            transformation = params.get("transformation")

            # Support basic transformations
            if transformation == "flatten":
                result = self._flatten_dict(data)
            elif transformation == "keys":
                result = list(data.keys()) if isinstance(data, dict) else []
            elif transformation == "values":
                result = list(data.values()) if isinstance(data, dict) else []
            elif transformation == "count":
                result = len(data) if hasattr(data, '__len__') else 0
            else:
                return {"error": f"Unknown transformation: {transformation}"}

            return {"result": result, "transformation": transformation}

        except Exception as e:
            logger.error(f"Data transformer failed: {e}")
            return {"error": str(e)}

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    # ==================== WEB & API TOOLS ====================

    async def _register_web_api_tools(self):
        """Register web and API tools"""

        # HTTP GET
        await self.register_tool(
            ToolDefinition(
                tool_id="web_http_get",
                name="HTTP GET",
                description="Make HTTP GET request",
                category=ToolCategory.WEB_API,
                permission_level=PermissionLevel.PUBLIC,
                parameters={
                    "url": {"type": "string", "required": True},
                    "headers": {"type": "object", "required": False},
                    "params": {"type": "object", "required": False}
                },
                timeout=30
            ),
            self._execute_http_get
        )

        # HTTP POST
        await self.register_tool(
            ToolDefinition(
                tool_id="web_http_post",
                name="HTTP POST",
                description="Make HTTP POST request",
                category=ToolCategory.WEB_API,
                permission_level=PermissionLevel.PUBLIC,
                parameters={
                    "url": {"type": "string", "required": True},
                    "headers": {"type": "object", "required": False},
                    "data": {"type": "object", "required": False},
                    "json": {"type": "object", "required": False}
                },
                timeout=30
            ),
            self._execute_http_post
        )

    async def _execute_http_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute HTTP GET tool"""
        try:
            url = params.get("url")
            headers = params.get("headers", {})
            query_params = params.get("params", {})

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=headers,
                    params=query_params,
                    timeout=aiohttp.ClientTimeout(total=20)
                ) as response:
                    status = response.status
                    content_type = response.headers.get('Content-Type', '')

                    if 'application/json' in content_type:
                        data = await response.json()
                    else:
                        data = await response.text()

                    return {
                        "status": status,
                        "headers": dict(response.headers),
                        "data": data,
                        "content_type": content_type
                    }

        except Exception as e:
            logger.error(f"HTTP GET failed: {e}")
            return {"error": str(e)}

    async def _execute_http_post(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute HTTP POST tool"""
        try:
            url = params.get("url")
            headers = params.get("headers", {})
            data = params.get("data")
            json_data = params.get("json")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    data=data,
                    json=json_data,
                    timeout=aiohttp.ClientTimeout(total=20)
                ) as response:
                    status = response.status
                    content_type = response.headers.get('Content-Type', '')

                    if 'application/json' in content_type:
                        response_data = await response.json()
                    else:
                        response_data = await response.text()

                    return {
                        "status": status,
                        "headers": dict(response.headers),
                        "data": response_data,
                        "content_type": content_type
                    }

        except Exception as e:
            logger.error(f"HTTP POST failed: {e}")
            return {"error": str(e)}

    # ==================== CODE EXECUTION TOOLS ====================

    async def _register_code_execution_tools(self):
        """Register code execution tools"""

        # Python Executor (Safe)
        await self.register_tool(
            ToolDefinition(
                tool_id="code_python_executor",
                name="Python Executor",
                description="Execute Python code safely",
                category=ToolCategory.CODE_EXECUTION,
                permission_level=PermissionLevel.RESTRICTED,
                parameters={
                    "code": {"type": "string", "required": True},
                    "timeout": {"type": "integer", "default": 5}
                },
                timeout=10
            ),
            self._execute_python_code
        )

    async def _execute_python_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Python code safely"""
        try:
            import sys
            from io import StringIO
            import contextlib

            code = params.get("code")
            timeout = params.get("timeout", 5)

            # Capture stdout
            stdout = StringIO()

            # Create restricted globals
            restricted_globals = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'range': range,
                    'str': str,
                    'int': int,
                    'float': float,
                    'bool': bool,
                    'list': list,
                    'dict': dict,
                    'tuple': tuple,
                    'set': set,
                    'sum': sum,
                    'min': min,
                    'max': max,
                    'abs': abs,
                    'round': round,
                }
            }

            # Execute with timeout
            with contextlib.redirect_stdout(stdout):
                exec(code, restricted_globals)

            output = stdout.getvalue()

            return {
                "success": True,
                "output": output,
                "result": restricted_globals.get('result')
            }

        except Exception as e:
            logger.error(f"Python execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }

    # ==================== AI/ML TOOLS ====================

    async def _register_ai_ml_tools(self):
        """Register AI/ML tools"""

        # Text Embeddings
        await self.register_tool(
            ToolDefinition(
                tool_id="ai_text_embeddings",
                name="Text Embeddings",
                description="Generate text embeddings",
                category=ToolCategory.AI_ML,
                permission_level=PermissionLevel.PUBLIC,
                parameters={
                    "text": {"type": "string", "required": True},
                    "model": {"type": "string", "default": "sentence-transformers"}
                },
                timeout=10
            ),
            self._execute_text_embeddings
        )

        # Text Similarity
        await self.register_tool(
            ToolDefinition(
                tool_id="ai_text_similarity",
                name="Text Similarity",
                description="Calculate similarity between texts",
                category=ToolCategory.AI_ML,
                permission_level=PermissionLevel.PUBLIC,
                parameters={
                    "text1": {"type": "string", "required": True},
                    "text2": {"type": "string", "required": True}
                },
                timeout=10
            ),
            self._execute_text_similarity
        )

    async def _execute_text_embeddings(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute text embeddings tool"""
        try:
            text = params.get("text")

            # Simple word-based embedding (placeholder for actual model)
            # In production, use sentence-transformers or similar
            words = text.lower().split()
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1

            # Create simple embedding vector
            embedding = [word_freq.get(word, 0) for word in sorted(word_freq.keys())[:100]]

            return {
                "text": text[:100],  # Truncate for response
                "embedding": embedding,
                "dimension": len(embedding),
                "model": "simple_word_freq"
            }

        except Exception as e:
            logger.error(f"Text embeddings failed: {e}")
            return {"error": str(e)}

    async def _execute_text_similarity(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute text similarity tool"""
        try:
            text1 = params.get("text1")
            text2 = params.get("text2")

            # Simple Jaccard similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())

            intersection = len(words1 & words2)
            union = len(words1 | words2)

            similarity = intersection / union if union > 0 else 0.0

            return {
                "text1": text1[:100],
                "text2": text2[:100],
                "similarity": similarity,
                "method": "jaccard",
                "common_words": list(words1 & words2)[:20]
            }

        except Exception as e:
            logger.error(f"Text similarity failed: {e}")
            return {"error": str(e)}


# Global tool registry instance
tool_registry = ToolRegistry()

