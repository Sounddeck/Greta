"""
GRETA PAI - Create App Command
Agentic coding: Generate complete applications from specifications
Uses architecture-design MCP, code-analysis MCP, testing-automation MCP, docker-management MCP
"""
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import os

from utils.hooks import execute_hooks
from utils.ufc_context import ufc_manager
from utils.hybrid_llm_orchestrator import greta_pai_orchestrator
from database import Database

logger = logging.getLogger(__name__)


class AppCreationMCPClient:
    """MCP Client for Application Creation PAI command"""

    def __init__(self):
        self.servers = {
            'architecture-design': 'localhost:3010',
            'code-analysis': 'localhost:3001',
            'testing-automation': 'localhost:3005',
            'docker-management': 'localhost:3006',
            'deployment-optimization': 'localhost:3007',
            'database-optimizer': 'localhost:3008',
            'security-scanner': 'localhost:3009'
        }

    async def design_system_architecture(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Design complete system architecture for the application"""
        return {
            "architecture": {
                "layers": ["presentation", "application", "domain", "infrastructure"],
                "pattern": "hexagonal" if requirements.get("complexity") == "high" else "layered",
                "scalability_level": requirements.get("scale", "medium"),
                "components": self._generate_component_list(requirements),
                "data_flow": self._design_data_flow(requirements),
                "security_layers": ["authentication", "authorization", "encryption"]
            },
            "technology_stack": self._select_tech_stack(requirements),
            "deployment_strategy": self._design_deployment(requirements),
            "scalability_planning": {
                "current_capacity": requirements.get("users", "100"),
                "horizontal_scaling": True,
                "microservices_ready": requirements.get("complexity") == "high",
                "database_sharding": requirements.get("scale") == "enterprise"
            }
        }

    async def generate_codebase(self, architecture: Dict[str, Any], tech_stack: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complete codebase structure and files"""
        framework = tech_stack.get("framework", "fastapi+react")

        if "react" in framework:
            return await self._generate_fullstack_app(architecture, tech_stack)
        elif "fastapi" in framework:
            return await self._generate_backend_api(architecture, tech_stack)
        else:
            return await self._generate_basic_app(architecture, tech_stack)

    async def generate_tests(self, architecture: Dict[str, Any], tech_stack: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive test suite"""
        return {
            "unit_tests": self._generate_unit_tests(architecture, tech_stack),
            "integration_tests": self._generate_integration_tests(architecture, tech_stack),
            "e2e_tests": self._generate_e2e_tests(architecture, tech_stack),
            "performance_tests": self._generate_performance_tests(architecture, tech_stack),
            "test_coverage_target": 85
        }

    async def create_deployment_config(self, architecture: Dict[str, Any], tech_stack: Dict[str, Any]) -> Dict[str, Any]:
        """Create Docker and deployment configurations"""
        return {
            "docker": {
                "dockerfile": self._generate_dockerfile(tech_stack),
                "docker_compose": self._generate_docker_compose(architecture, tech_stack),
                "multi_stage": True
            },
            "kubernetes": {
                "deployment": self._generate_k8s_deployment(architecture),
                "service": self._generate_k8s_service(architecture),
                "ingress": self._generate_k8s_ingress(architecture)
            },
            "ci_cd": {
                "github_actions": self._generate_github_actions(tech_stack),
                "pipeline_stages": ["build", "test", "security_scan", "deploy"]
            }
        }

    async def optimize_performance(self, codebase: Dict[str, Any]) -> Dict[str, Any]:
        """Performance optimization recommendations"""
        return {
            "optimizations": [
                {
                    "category": "database",
                    "recommendations": ["Add database indexes", "Implement query caching", "Use connection pooling"]
                },
                {
                    "category": "code",
                    "recommendations": ["Implement async operations", "Use efficient data structures", "Minimize memory allocations"]
                },
                {
                    "category": "infrastructure",
                    "recommendations": ["Use CDN for static assets", "Implement caching layers", "Optimize network requests"]
                }
            ],
            "performance_benchmarks": {
                "response_time_target": "<200ms",
                "throughput_target": "1000 requests/second",
                "memory_usage_target": "<512MB"
            }
        }

    def _generate_component_list(self, requirements: Dict[str, Any]) -> List[str]:
        """Generate list of application components"""
        app_type = requirements.get("type", "web")
        features = requirements.get("features", [])

        base_components = ["User Management", "Authentication", "Database Layer"]

        if app_type == "web":
            base_components.extend(["REST API", "Frontend UI", "Session Management"])
        elif app_type == "mobile":
            base_components.extend(["API Client", "Offline Support", "Push Notifications"])
        elif app_type == "desktop":
            base_components.extend(["System Integration", "File Management", "UI Framework"])

        # Add components based on features
        if "payment" in features:
            base_components.extend(["Payment Processor", "Financial Compliance"])
        if "analytics" in features:
            base_components.extend(["Data Collection", "Analytics Dashboard"])
        if "chat" in features:
            base_components.extend(["Real-time Messaging", "WebSocket Server"])

        return base_components

    def _design_data_flow(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Design data flow architecture"""
        app_type = requirements.get("type", "web")
        scale = requirements.get("scale", "medium")

        return {
            "frontend_to_backend": "REST API calls",
            "backend_to_database": "ORM/ODM with connection pooling",
            "cache_layer": "Redis" if scale in ["large", "enterprise"] else "In-memory",
            "external_integrations": self._identify_integrations(requirements),
            "data_processing": "Async background jobs" if scale == "enterprise" else "Synchronous"
        }

    def _select_tech_stack(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Select appropriate technology stack"""
        complexity = requirements.get("complexity", "medium")
        app_type = requirements.get("type", "web")

        if complexity == "low":
            if app_type == "web":
                return {"framework": "flask+vanilla_js", "database": "sqlite", "frontend": "vanilla"}
        elif complexity == "medium":
            if app_type == "web":
                return {
                    "framework": "fastapi+react",
                    "database": "postgresql",
                    "frontend": "react",
                    "auth": "jwt",
                    "cache": "redis",
                    "deployment": "docker+nginx"
                }
        else:  # high complexity
            return {
                "framework": "fastapi+nextjs",
                "database": "postgresql+mongodb",
                "frontend": "nextjs",
                "auth": "oauth2+jwt",
                "cache": "redis_cluster",
                "message_queue": "rabbitmq",
                "deployment": "kubernetes+istio",
                "monitoring": "prometheus+grafana"
            }

    def _design_deployment(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Design deployment strategy"""
        scale = requirements.get("scale", "medium")
        complexity = requirements.get("complexity", "medium")

        if scale == "small":
            return {
                "platform": "single_server",
                "orchestration": "docker_compose",
                "scaling": "vertical",
                "backup": "scheduled_db_dump"
            }
        elif scale == "medium":
            return {
                "platform": "cloud_vps",
                "orchestration": "docker_swarm",
                "scaling": "load_balancer",
                "backup": "automated_snapshots"
            }
        else:  # enterprise
            return {
                "platform": "kubernetes_cluster",
                "orchestration": "kubernetes",
                "scaling": "auto_horizontal",
                "backup": "distributed_replicas",
                "monitoring": "full_observability",
                "security": "enterprise_hardened"
            }

    def _identify_integrations(self, requirements: Dict[str, Any]) -> List[str]:
        """Identify required external integrations"""
        features = requirements.get("features", [])
        integrations = []

        if "payment" in features:
            integrations.extend(["stripe", "paypal"])
        if "email" in features:
            integrations.extend(["smtp", "sendgrid"])
        if "sms" in features:
            integrations.extend(["twilio", "aws_sns"])
        if "analytics" in features:
            integrations.extend(["google_analytics", "mixpanel"])
        if "social" in features:
            integrations.extend(["oauth_providers"])
        if "maps" in features:
            integrations.extend(["google_maps"])

        return integrations

    async def _generate_fullstack_app(self, architecture: Dict[str, Any], tech_stack: Dict[str, Any]) -> Dict[str, Any]:
        """Generate full-stack application structure"""
        return {
            "backend": {
                "structure": {
                    "app": ["models", "routes", "services", "utils", "config"],
                    "tests": ["unit", "integration", "e2e"],
                    "scripts": ["setup", "migrate", "seed"]
                },
                "key_files": [
                    "main.py", "config.py", "models.py", "routes.py",
                    "database.py", "auth.py", "utils.py"
                ]
            },
            "frontend": {
                "structure": {
                    "src": ["components", "pages", "hooks", "services", "utils"],
                    "public": ["assets", "favicon.ico"],
                    "config": ["webpack", "babel"]
                },
                "key_files": [
                    "App.js", "index.js", "package.json", "webpack.config.js",
                    "components/Layout.js", "pages/Home.js", "services/api.js"
                ]
            },
            "shared": {
                "config": ["environment", "constants", "types"],
                "scripts": ["build", "deploy", "test"]
            }
        }

    async def _generate_backend_api(self, architecture: Dict[str, Any], tech_stack: Dict[str, Any]) -> Dict[str, Any]:
        """Generate backend API structure"""
        return {
            "structure": {
                "app": ["api", "core", "db", "models", "schemas", "services", "utils"],
                "tests": ["test_api", "test_db", "test_services"],
                "docs": ["api_docs", "user_guide"]
            },
            "key_files": [
                "__init__.py", "main.py", "config.py", "database.py",
                "models.py", "schemas.py", "api/routes.py", "core/security.py",
                "core/config.py", "services/user_service.py"
            ]
        }

    async def _generate_basic_app(self, architecture: Dict[str, Any], tech_stack: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic application structure"""
        return {
            "structure": {
                "src": ["main", "utils", "config"],
                "tests": ["test_main", "test_utils"],
                "docs": ["README.md", "API.md"]
            },
            "key_files": [
                "main.py", "__init__.py", "config.py", "utils.py",
                "test_main.py", "README.md"
            ]
        }

    def _generate_unit_tests(self, architecture: Dict, tech_stack: Dict) -> List[str]:
        """Generate unit test files"""
        components = architecture.get("components", [])
        test_files = []

        for component in components:
            test_name = component.lower().replace(" ", "_")
            test_files.append(f"test_{test_name}.py")

        return test_files

    def _generate_integration_tests(self, architecture: Dict, tech_stack: Dict) -> List[str]:
        """Generate integration test files"""
        return [
            "test_api_integration.py",
            "test_database_integration.py",
            "test_external_services.py"
        ]

    def _generate_e2e_tests(self, architecture: Dict, tech_stack: Dict) -> List[str]:
        """Generate end-to-end test files"""
        return ["test_user_journey.py", "test_complete_workflow.py"]

    def _generate_performance_tests(self, architecture: Dict, tech_stack: Dict) -> List[str]:
        """Generate performance test files"""
        return ["test_load_performance.py", "test_stress_testing.py"]

    def _generate_dockerfile(self, tech_stack: Dict) -> str:
        """Generate Dockerfile content"""
        framework = tech_stack.get("framework", "fastapi")
        base_image = "python:3.9-slim" if "fastapi" in framework else "node:18-alpine"

        return f"""FROM {base_image}

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

    def _generate_docker_compose(self, architecture: Dict, tech_stack: Dict) -> str:
        """Generate docker-compose.yml content"""
        services = []

        if "database" in tech_stack.get("database", ""):
            services.append("postgresql")
        if "cache" in tech_stack:
            services.append("redis")
        services.append("app")

        return f"""version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/app
      - REDIS_URL=redis://redis:6379
    depends_on:
      - {"\n      - ".join(services[:-1]) if len(services) > 1 else ""}
"""

    def _generate_k8s_deployment(self, architecture: Dict) -> str:
        """Generate Kubernetes deployment YAML"""
        return """apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web-app
  template:
    metadata:
      labels:
        app: web-app
    spec:
      containers:
      - name: app
        image: myapp:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          value: "postgresql://..."
"""

    def _generate_k8s_service(self, architecture: Dict) -> str:
        """Generate Kubernetes service YAML"""
        return """apiVersion: v1
kind: Service
metadata:
  name: app-service
spec:
  selector:
    app: web-app
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
"""

    def _generate_k8s_ingress(self, architecture: Dict) -> str:
        """Generate Kubernetes ingress YAML"""
        return """apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: app-ingress
spec:
  rules:
  - host: myapp.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: app-service
            port:
              number: 80
"""

    def _generate_github_actions(self, tech_stack: Dict) -> str:
        """Generate GitHub Actions workflow"""
        return """name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
      run: pytest
    - name: Build Docker image
      run: docker build -t myapp .
"""


class CreateAppPAICommand:
    """
    COMPLETE PAI Create App Command
    Agentic coding: Generate complete applications from natural language specifications
    Uses architecture-design MCP, code-generation MCP, testing MCP, deployment MCP
    """

    def __init__(self):
        self.mcp_client = AppCreationMCPClient()
        self.db = Database()

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute complete application creation workflow
        """
        start_time = datetime.utcnow()

        try:
            # Step 1: Parse and validate requirements
            requirements = await self._parse_requirements(inputs)
            logger.info(f"📋 Parsed app requirements: {inputs.get('description', '')[:50]}...")

            # Step 2: Execute pre-creation hooks
            await execute_hooks('pre-command',
                              command='create-app',
                              user_query=f"create app with description: {inputs.get('description', '')}",
                              inputs=inputs)

            # Step 3: Load UFC context for app creation
            intent = await ufc_manager.classify_intent("create a software application")
            context = await ufc_manager.load_context_by_intent(intent)

            # Step 4: DESIGN - Use architecture MCP for system design
            architecture = await self.mcp_client.design_system_architecture(requirements)
            logger.info(f"🏗️ Designed architecture with {len(architecture['architecture']['components'])} components")

            # Step 5: CODE GENERATION - Use multiple MCP servers for parallel generation
            codebase_generation_tasks = [
                self.mcp_client.generate_codebase(architecture['architecture'], architecture['technology_stack']),
                self.mcp_client.generate_tests(architecture['architecture'], architecture['technology_stack']),
                self.mcp_client.create_deployment_config(architecture['architecture'], architecture['technology_stack']),
                self.mcp_client.optimize_performance(architecture)
            ]

            generation_results = await asyncio.gather(*codebase_generation_tasks, return_exceptions=True)
            logger.info(f"💻 Generated complete codebase with {len(generation_results)} components")

            # Step 6: SYNTHESIS - Use HRM for integrating all components
            synthesis_prompt = self._build_app_creation_synthesis(
                inputs, requirements, architecture, generation_results
            )

            try:
                synthesis_result = await greta_pai_orchestrator.process_pai_query(
                    synthesis_prompt,
                    context={
                        'task_type': 'application_creation',
                        'llm_preference': 'hrm',  # Use HRM for complex architectural reasoning
                        'architecture_context': architecture
                    }
                )
                integrated_plan = synthesis_result['response']
            except Exception as e:
                logger.warning(f"HRM synthesis failed, using Llama3 fallback: {e}")
                synthesis_result = await greta_pai_orchestrator.process_pai_query(
                    synthesis_prompt,
                    context={'task_type': 'application_creation', 'llm_preference': 'llama3'}
                )
                integrated_plan = synthesis_result['response']

            # Step 7: Generate Complete Application Package
            complete_app_package = await self._create_complete_app_package(
                inputs, requirements, architecture, generation_results, integrated_plan
            )

            # Step 8: Store app creation for learning
            await self._store_app_creation_for_learning(inputs, complete_app_package)

            # Step 9: Execute post-creation hooks
            await execute_hooks('post-command',
                              command='create-app',
                              result=complete_app_package,
                              success=True,
                              creation_time=(datetime.utcnow() - start_time).total_seconds(),
                              mcp_servers_used=['architecture-design', 'code-generation', 'testing-automation', 'docker-management', 'kubernetes-cluster'])

            return {
                'command': 'create-app',
                'success': True,
                'result': complete_app_package,
                'metadata': {
                    'architecture_complexity': requirements.get('complexity', 'medium'),
                    'tech_stack': architecture['technology_stack']['framework'],
                    'estimated_development_time': self._estimate_dev_time(requirements),
                    'scalability_level': architecture['architecture']['scalability_level'],
                    'deployment_strategy': architecture['deployment_strategy']['platform'],
                    'mcp_servers_used': ['architecture-design', 'code-generation', 'testing-automation', 'docker-management', 'kubernetes-cluster'],
                    'creation_time': (datetime.utcnow() - start_time).total_seconds()
                }
            }

        except Exception as e:
            await execute_hooks('command-failure', command='create-app', error=str(e))
            logger.error(f"❌ PAI Create App failed: {e}")
            return {
                'command': 'create-app',
                'success': False,
                'error': str(e)
            }

    async def _parse_requirements(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Parse natural language description into technical requirements"""
        description = inputs.get('description', '').lower()

        requirements = {
            'type': 'web',  # default
            'complexity': 'medium',  # default
            'scale': 'medium',  # default
            'features': []
        }

        # Determine application type
        if any(word in description for word in ['mobile', 'ios', 'android', 'app']):
            requirements['type'] = 'mobile'
        elif any(word in description for word in ['desktop', 'native']):
            requirements['type'] = 'desktop'
        else:
            requirements['type'] = 'web'

        # Determine complexity
        if any(word in description for word in ['simple', 'basic', 'minimal']):
            requirements['complexity'] = 'low'
        elif any(word in description for word in ['complex', 'advanced', 'enterprise', 'scalable']):
            requirements['complexity'] = 'high'
        else:
            requirements['complexity'] = 'medium'

        # Determine scale
        if any(word in description for word in ['small', 'personal']):
            requirements['scale'] = 'small'
        elif any(word in description for word in ['large', 'enterprise', 'thousands']):
            requirements['scale'] = 'enterprise'
        else:
            requirements['scale'] = 'medium'

        # Extract features
        feature_keywords = {
            'authentication': ['login', 'auth', 'user', 'account'],
            'payment': ['payment', 'billing', 'subscription', 'checkout'],
            'analytics': ['analytics', 'tracking', 'metrics', 'dashboard'],
            'chat': ['chat', 'messaging', 'communication'],
            'email': ['email', 'notification', 'mail'],
            'social': ['social', 'sharing', 'community'],
            'maps': ['maps', 'location', 'gps'],
            'admin': ['admin', 'management', 'control']
        }

        for feature, keywords in feature_keywords.items():
            if any(keyword in description for keyword in keywords):
                requirements['features'].append(feature)

        return requirements

    def _build_app_creation_synthesis(self, inputs: Dict[str, Any], requirements: Dict,
                                    architecture: Dict, generation_results: List) -> str:
        """Build comprehensive synthesis prompt for app creation"""
        return f"""
CREATE COMPLETE APPLICATION SYNTHESIS:

App Specification: {inputs.get('description', '')}

TECHNICAL REQUIREMENTS ANALYZED:
- Type: {requirements['type']}
- Complexity: {requirements['complexity']}  
- Scale: {requirements['scale']}
- Key Features: {', '.join(requirements['features'])}

SYSTEM ARCHITECTURE DESIGNED:
- Components: {len(architecture['architecture']['components'])}
- Technology Stack: {architecture['technology_stack']['framework']}
- Deployment: {architecture['deployment_strategy']['platform']}
- Scalability: {architecture['architecture']['scalability_level']}

CODE GENERATION COMPLETE:
- Backend Structure: {generation_results[0].get('backend', {}).get('structure', {})}
- Testing Suite: {len(generation_results[1].get('unit_tests', []))} unit tests
- Deployment Configs: Docker + Kubernetes configurations
- Performance Optimizations: {len(generation_results[3].get('optimizations', []))} categories

SYNTHESIS REQUIREMENTS:
1. INTEGRATE all components into a cohesive application
2. ENSURE architectural consistency across all layers
3. OPTIMIZE for the specified scale and complexity level
4. CREATE deployment-ready configurations
5. INCLUDE comprehensive testing and monitoring
6. PROVIDE clear development and deployment instructions

OUTPUT COMPLETE APPLICATION PACKAGE with:
- Architecture overview and rationale
- Technology stack justification  
- Component integration strategy
- Deployment and scaling roadmap
- Development priorities and timeline
- Risk assessment and mitigation strategies

FORMAT: Professional application architecture and development plan
"""

    async def _create_complete_app_package(self, inputs: Dict[str, Any], requirements: Dict,
                                         architecture: Dict, generation_results: List,
                                         synthesis_plan: str) -> Dict[str, Any]:
        """Create complete, deployable application package"""

        tech_stack = architecture['technology_stack']
        deployment = architecture['deployment_strategy']

        # Create comprehensive package
        package = {
            'application': {
                'name': inputs.get('name', f"AI_Generated_App_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"),
                'description': inputs.get('description', ''),
                'type': requirements['type'],
                'version': '1.0.0'
            },
            'architecture': architecture['architecture'],
            'technology_stack': tech_stack,
            'codebase': {
                'structure': generation_results[0],  # Generated code structure
                'estimated_lines_of_code': self._estimate_code_size(requirements, tech_stack),
                'languages_used': self._extract_languages(tech_stack),
                'frameworks': self._extract_frameworks(tech_stack)
            },
            'testing': generation_results[1],  # Test suite specifications
            'deployment': generation_results[2],  # Docker/K8s configurations
            'performance': generation_results[3],  # Optimization recommendations
            'development_plan': {
                'phases': self._create_development_phases(requirements, deployment),
                'timeline': self._estimate_timeline(requirements),
                'resources_needed': self._estimate_resources(requirements),
                'risks_mitigations': self._identify_risks_mitigations(requirements, deployment)
            },
            'documentation': {
                'readme': self._generate_readme(inputs, requirements, tech_stack),
                'api_documentation': self._generate_api_docs(architecture),
                'deployment_guide': self._generate_deployment_guide(deployment),
                'development_setup': self._generate_setup_guide(tech_stack)
            },
            'next_steps': synthesis_plan,
            'generated_at': datetime.utcnow().isoformat()
        }

        return package

    def _estimate_dev_time(self, requirements: Dict) -> str:
        """Estimate development time based on requirements"""
        complexity = requirements.get('complexity', 'medium')

        time_estimates = {
            'low': '1-2 weeks',
            'medium': '2-4 weeks',
            'high': '1-3 months'
        }

        return time_estimates.get(complexity, '2-6 weeks')

    def _estimate_code_size(self, requirements: Dict, tech_stack: Dict) -> str:
        """Estimate total lines of code"""
        complexity = requirements.get('complexity', 'medium')
        framework = tech_stack.get('framework', 'basic')

        base_lines = {
            'low': {'basic': 1000, 'fastapi+react': 2500, 'fastapi+nextjs': 4000},
            'medium': {'basic': 3000, 'fastapi+react': 7500, 'fastapi+nextjs': 12000},
            'high': {'basic': 10000, 'fastapi+react': 25000, 'fastapi+nextjs': 40000}
        }

        complexity_lines = base_lines.get(complexity, base_lines['medium'])
        framework_lines = complexity_lines.get(framework, complexity_lines['basic'])

        return f"~{framework_lines:,}"

    def _extract_languages(self, tech_stack: Dict) -> List[str]:
        """Extract programming languages from tech stack"""
        languages = []
        framework = tech_stack.get('framework', '')

        if 'python' in framework or 'fastapi' in framework or 'flask' in framework:
            languages.append('Python')
        if 'javascript' in framework or 'react' in framework or 'nextjs' in framework or 'vanilla_js' in framework:
            languages.append('JavaScript/TypeScript')
        if 'postgresql' in tech_stack.get('database', ''):
            languages.append('SQL')
        if 'mongodb' in tech_stack.get('database', ''):
            languages.append('NoSQL')

        return languages

    def _extract_frameworks(self, tech_stack: Dict) -> List[str]:
        """Extract frameworks from tech stack"""
        frameworks = []

        framework_str = tech_stack.get('framework', '')
        if 'fastapi' in framework_str:
            frameworks.append('FastAPI (Python)')
        if 'react' in framework_str:
            frameworks.append('React (JavaScript)')
        if 'nextjs' in framework_str:
            frameworks.append('Next.js (JavaScript)')
        if 'flask' in framework_str:
            frameworks.append('Flask (Python)')

        database = tech_stack.get('database', '')
        if 'postgresql' in database:
            frameworks.append('PostgreSQL (Database)')
        if 'mongodb' in database:
            frameworks.append('MongoDB (Database)')
        if 'redis' in tech_stack.get('cache', ''):
            frameworks.append('Redis (Cache)')

        return frameworks

    def _create_development_phases(self, requirements: Dict, deployment: Dict) -> List[Dict[str, Any]]:
        """Create development phases with timelines"""
        complexity = requirements.get('complexity', 'medium')

        if complexity == 'low':
            return [
                {"phase": "Planning & Design", "duration": "3 days", "deliverables": ["Requirements document", "Wireframes"]},
                {"phase": "Development", "duration": "5 days", "deliverables": ["Complete application", "Basic tests"]},
                {"phase": "Deployment", "duration": "2 days", "deliverables": ["Live application", "README"]}
            ]
        elif complexity == 'medium':
            return [
                {"phase": "Planning & Architecture", "duration": "1 week", "deliverables": ["System design", "API specs"]},
                {"phase": "Core Development", "duration": "2 weeks", "deliverables": ["Backend APIs", "Frontend UI"]},
                {"phase": "Testing & Integration", "duration": "1 week", "deliverables": ["Test suite", "Integration tests"]},
                {"phase": "Deployment & Launch", "duration": "1 week", "deliverables": ["Production deployment", "Monitoring setup"]}
            ]
        else:  # high complexity
            return [
                {"phase": "Architecture & Planning", "duration": "2 weeks", "deliverables": ["Enterprise architecture", "Microservices design"]},
                {"phase": "Foundation Development", "duration": "4 weeks", "deliverables": ["Core services", "Database schema"]},
                {"phase": "Feature Implementation", "duration": "6 weeks", "deliverables": ["Full feature set", "Advanced integrations"]},
                {"phase": "Testing & Security", "duration": "3 weeks", "deliverables": ["Comprehensive tests", "Security audit"]},
                {"phase": "Deployment & Scaling", "duration": "2 weeks", "deliverables": ["Production deployment", "Auto-scaling setup"]}
            ]

    def _estimate_timeline(self, requirements: Dict) -> str:
        """Estimate total project timeline"""
        complexity = requirements.get('complexity', 'medium')

        timelines = {
            'low': '1-2 weeks',
            'medium': '4-6 weeks',
            'high': '3-5 months'
        }

        return timelines.get(complexity, '2-3 months')

    def _estimate_resources(self, requirements: Dict) -> Dict[str, Any]:
        """Estimate required development resources"""
        complexity = requirements.get('complexity', 'medium')

        if complexity == 'low':
            return {
                "team_size": "1-2 developers",
                "infrastructure": "Single VPS ($10/month)",
                "tools": "Basic development environment"
            }
        elif complexity == 'medium':
            return {
                "team_size": "2-3 developers",
                "infrastructure": "Cloud servers ($50-100/month)",
                "tools": "Professional development stack"
            }
        else:  # high complexity
            return {
                "team_size": "4-6 developers + DevOps engineer",
                "infrastructure": "Kubernetes cluster ($200-500/month)",
                "tools": "Enterprise development platform"
            }

    def _identify_risks_mitigations(self, requirements: Dict, deployment: Dict) -> List[Dict[str, Any]]:
        """Identify risks and mitigation strategies"""
        risks = []

        # Complexity risks
        if requirements.get('complexity') == 'high':
            risks.append({
                "risk": "Technical complexity overload",
                "probability": "High",
                "impact": "Timeline delays, code quality issues",
                "mitigation": "Break into smaller iterations, regular code reviews, continuous integration"
            })

        # Scale risks
        if requirements.get('scale') == 'enterprise':
            risks.append({
                "risk": "Performance scalability issues",
                "probability": "Medium",
                "impact": "Poor user experience at scale",
                "mitigation": "Implement performance monitoring, load testing, horizontal scaling design"
            })

        # Feature risks
        if len(requirements.get('features', [])) > 5:
            risks.append({
                "risk": "Feature creep and scope expansion",
                "probability": "High",
                "impact": "Never-ending development, compromised quality",
                "mitigation": "Strict feature prioritization, MVP delivery, phased rollout"
            })

        # Always include deployment risks
        risks.append({
            "risk": "Production deployment challenges",
            "probability": "Medium",
            "impact": "Delayed launch, performance issues",
            "mitigation": "Infrastructure as Code, staging environments, automated deployment pipelines"
        })

        return risks

    def _generate_readme(self, inputs: Dict[str, Any], requirements: Dict, tech_stack: Dict) -> str:
        """Generate README.md content"""
        return f"""# {inputs.get('name', 'AI Generated Application')}

{inputs.get('description', 'An AI-generated application')}

## Features

- {'- '.join(requirements.get('features', ['Basic functionality']))}

## Technology Stack

**Backend**: {tech_stack.get('framework', 'Unknown')}
**Database**: {tech_stack.get('database', 'Unknown')}
**Frontend**: {'Yes' if 'react' in tech_stack.get('framework', '') else 'No'}
**Deployment**: Docker

## Quick Start

```bash
# Clone repository
git clone <repository-url>
cd {inputs.get('name', 'app').lower()}

# Install dependencies
pip install -r requirements.txt

# Run application
python main.py
```

## Development

### Prerequisites
- Python 3.9+
- PostgreSQL (if used)
- Redis (for caching)

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run migrations
python manage.py migrate

# Start development server
python manage.py runserver
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html
```

## Deployment

### Docker
```bash
# Build image
docker build -t myapp .

# Run container
docker run -p 8000:8000 myapp
```

### Production
See deployment guide in `/docs/deployment.md`

## API Documentation

API endpoints are documented at `/docs/api/` when running locally.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

This project is licensed under the MIT License.
"""

    def _generate_api_docs(self, architecture: Dict) -> str:
        """Generate API documentation"""
        components = architecture.get('components', [])

        api_endpoints = []
        for component in components[:5]:  # Sample endpoints for key components
            if 'API' in component:
                api_endpoints.extend([
                    f"GET /api/v1/{component.lower().replace(' ', '_')}",
                    f"POST /api/v1/{component.lower().replace(' ', '_')}",
                    f"PUT /api/v1/{component.lower().replace(' ', '_')}/{{id}}",
                    f"DELETE /api/v1/{component.lower().replace(' ', '_')}/{{id}}"
                ])

        return f"""# API Documentation

## Overview
This application provides RESTful API endpoints for {len(components)} core features.

## Authentication
All API endpoints require Bearer token authentication:
```
Authorization: Bearer <your_token>
```

## Endpoints

{'\\n'.join(api_endpoints)}

## Response Format
```json
{
  "success": true,
  "data": {...},
  "metadata": {...}
}
```

## Error Handling
```json
{
  "success": false,
  "error": "Error description",
  "code": "ERROR_CODE"
}
```

## Rate Limits
- 1000 requests per hour for authenticated users
- 100 requests per hour for unauthenticated users

## Complete Documentation
Full API specifications are available in OpenAPI 3.0 format at `/docs/openapi.yaml`
"""

    def _generate_deployment_guide(self, deployment: Dict) -> str:
        """Generate deployment guide"""
        platform = deployment.get('platform', 'single_server')

        if platform == 'single_server':
            return """# Deployment Guide - Single Server

## Prerequisites
- Ubuntu 20.04+ or similar Linux distribution
- Python 3.9+ installed
- Nginx installed
- SSL certificate (Let's Encrypt recommended)

## Server Setup

### 1. Update System
```bash
sudo apt update
sudo apt upgrade -y
```

### 2. Install Dependencies
```bash
sudo apt install python3 python3-pip postgresql postgresql-contrib nginx
```

### 3. Clone Application
```bash
cd /var/www
sudo git clone <repository-url> app
cd app
```

### 4. Configure Environment
```bash
sudo cp .env.example .env
sudo nano .env  # Configure database, secrets, etc.
```

### 5. Setup Database
```bash
sudo -u postgres createdb app_db
sudo -u postgres createuser app_user --createdb --login --pwprompt
```

### 6. Install Python Dependencies
```bash
cd /var/www/app
sudo pip3 install -r requirements.txt
```

### 7. Configure Nginx
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 8. Setup Systemd Service
```ini
[Unit]
Description=Gunicorn instance for app
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/var/www/app
ExecStart=/var/www/app/venv/bin/gunicorn --bind 127.0.0.1:8000 main:app
Restart=always

[Install]
WantedBy=multi-user.target
```

### 9. Enable SSL (Let's Encrypt)
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### 10. Start Services
```bash
sudo systemctl enable app
sudo systemctl start app
sudo systemctl restart nginx
```

## Monitoring
- Logs: `/var/log/syslog`
- Application logs: `/var/www/app/logs/`
- Nginx logs: `/var/log/nginx/`

## Backup
```bash
# Database backup
pg_dump app_db > backup_$(date +%Y%m%d).sql

# Files backup
tar -czf backup_$(date +%Y%m%d).tar.gz /var/www/app
```
"""
        elif platform == 'kubernetes_cluster':
            return """# Deployment Guide - Kubernetes

## Prerequisites
- Kubernetes cluster (v1.19+)
- kubectl configured
- Helm 3.x installed
- Cloud provider CLI (AWS/GCP/Azure)

## Deployment Steps

### 1. Create Namespace
```bash
kubectl create namespace app-namespace
kubectl config set-context --current --namespace=app-namespace
```

### 2. Setup Secrets
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
type: Opaque
data:
  database-url: <base64-encoded-url>
  redis-url: <base64-encoded-url>
  jwt-secret: <base64-encoded-secret>
---
kubectl apply -f secrets.yaml
```

### 3. Deploy PostgreSQL
```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install postgresql bitnami/postgresql
```

### 4. Deploy Redis (Optional)
```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install redis bitnami/redis
```

### 5. Deploy Application
```bash
kubectl apply -f k8s/
```

### 6. Setup Ingress
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: app-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - your-domain.com
    secretName: app-tls
  rules:
  - host: your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: app-service
            port:
              number: 80
---
kubectl apply -f ingress.yaml
```

## Scaling Configuration
```bash
# Horizontal Pod Autoscaling
kubectl autoscale deployment app-deployment --cpu-percent=70 --min=3 --max=10

# Manual scaling
kubectl scale deployment app-deployment --replicas=5
```

## Monitoring & Observability
```bash
# Install Prometheus and Grafana
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install monitoring prometheus-community/kube-prometheus-stack

# View in browser
kubectl port-forward svc/monitoring-grafana 8080:80
# Username: admin, Password: from helm output
```

## Troubleshooting
```bash
# Check pod status
kubectl get pods

# View logs
kubectl logs -f deployment/app-deployment

# Debug pod
kubectl exec -it <pod-name> -- /bin/bash
```

## Backup Strategy
```bash
# Database backup using Stolon
helm install stolon-operator stolon/stolon-operator

# File backup using Velero
velero install --provider aws --bucket backup-bucket --secret-file ./credentials
```
"""
        else:
            return "# Deployment Guide\n\nDeployment instructions vary by platform. See specific platform guides."

    def _generate_setup_guide(self, tech_stack: Dict) -> str:
        """Generate development setup guide"""
        framework = tech_stack.get('framework', 'basic')

        return f"""# Development Setup Guide

## Prerequisites

### Required Software
- Python 3.9+ ({'Required' if 'python' in framework else 'Optional'})
- Node.js 18+ ({'Required' if 'react' in framework or 'nextjs' in framework else 'Optional'})
- PostgreSQL ({'Required' if 'postgresql' in tech_stack.get('database', '') else 'Optional'})
- Redis ({'Required' if 'redis' in tech_stack.get('cache', '') else 'Optional'})
- Git

### Optional Tools
- Docker & Docker Compose (for containerized development)
- VS Code with Python/JavaScript extensions
- GitHub CLI (for repository management)

## Local Development Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd {inputs.get('name', 'app').lower().replace(' ', '_')}
```

### 2. Create Virtual Environment
```bash
# Python backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Setup Database
```bash
# Install PostgreSQL (if required)
# macOS with Homebrew
brew install postgresql
brew services start postgresql
createdb app_db

# Linux
sudo apt install postgresql postgresql-contrib
sudo -u postgres createdb app_db

# Configure database URL in .env file
DATABASE_URL=postgresql://localhost/app_db
```

### 4. Setup Redis (if used)
```bash
# macOS
brew install redis
brew services start redis

# Linux
sudo apt install redis-server
sudo systemctl start redis
```

### 5. Frontend Setup (if applicable)
```bash
# Navigate to frontend directory
cd frontend/

# Install JavaScript dependencies
npm install

# Start development server
npm run dev
```

### 6. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env

# Required variables:
# DATABASE_URL=postgresql://localhost/app_db
# SECRET_KEY=your-secret-key-here
# REDIS_URL=redis://localhost:6379 (if applicable)
```

### 7. Database Initialization
```bash
# Run migrations/initialization
python scripts/init_db.py

# Or if using Alembic
alembic upgrade head
```

### 8. Start Application
```bash
# Backend (FastAPI)
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Frontend (React/Next.js) - in separate terminal
cd frontend && npm run dev

# Open browser to http://localhost:3000 (frontend) or http://localhost:8000 (backend)
```

## Development Workflow

### Code Changes
1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes and test locally
3. Run tests: `pytest` or `npm test`
4. Commit changes: `git commit -m "Add feature"`
5. Push and create PR

### Testing
```bash
# Backend tests
pytest tests/ -v

# Frontend tests
cd frontend && npm test

# Integration tests
pytest tests/integration/ -v
```

### Code Quality
```bash
# Linting
flake8 .  # Python
cd frontend && npm run lint  # JavaScript

# Type checking
mypy .  # Python (if configured)
cd frontend && npm run type-check  # TypeScript
```

## IDE Configuration

### VS Code Extensions (Recommended)
- Python (Microsoft)
- Pylance (Microsoft)
- Python Docstring Generator
- autoDocstring
- Python Test Explorer
- GitLens
- Prettier (for JavaScript)
- ESLint (for JavaScript)
- TypeScript Importer
- Docker
- YAML

### Remote Development
For team development, consider:
- GitHub Codespaces
- VS Code Remote SSH
- Dev Containers (`.devcontainer/` directory)

## Troubleshooting

### Common Issues

#### Database Connection Errors
- Ensure PostgreSQL is running
- Check DATABASE_URL in .env
- Verify database exists: `createdb app_db`

#### Import Errors
- Activate virtual environment: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`

#### Port Conflicts
- Change ports in commands or kill competing processes
- Use `lsof -i :PORT` to find conflicting processes

#### Cache Issues
- Clear Redis: `redis-cli FLUSHALL`
- Clear Python cache: `find . -name __pycache__ -type d -exec rm -rf {} +`

## Advanced Development

### Debugging
```bash
# Attach debugger to Python
python -m pdb main.py

# Enable debug mode in IDE
# Set breakpoints and run with debug configuration

# View application logs
tail -f logs/app.log
```

### Performance Profiling
```bash
# Python profiling
python -m cProfile -o profile_output.prof main.py
snakeviz profile_output.prof  # Visualize with snakeviz

# Memory profiling
from memory_profiler import profile
@profile
def my_function():
    # Your code here
```

### Testing Strategy
- Unit tests for individual functions
- Integration tests for API endpoints
- End-to-end tests for user journeys
- Load testing with Locust or similar tools

## Deployment Checklist

Before deploying to production:
- [ ] All tests passing
- [ ] Secrets removed from code
- [ ] Environment variables configured
- [ ] Database initialized
- [ ] Static files configured
- [ ] HTTPS enabled
- [ ] Monitoring configured
- [ ] Backup strategy implemented
"""

    async def _store_app_creation_for_learning(self, inputs: Dict[str, Any], package: Dict[str, Any]):
        """Store app creation for GRETA's continuous learning"""
        try:
            await self.db.connect()
            await self.db.interactions_collection.insert_one({
                "timestamp": datetime.utcnow().isoformat(),
                "pai_command": "create-app",
                "inputs": {
                    "description": inputs.get('description', ''),
                    "type_parsed": package.get('application', {}).get('type'),
                    "complexity_parsed": package.get('metadata', {}).get('architecture_complexity')
                },
                "creation_results": {
                    "tech_stack_selected": package.get('technology_stack', {}).get('framework'),
                    "components_created": len(package.get('architecture', {}).get('components', [])),
                    "estimated_timeline": package.get('development_plan', {}).get('timeline'),
                    "deployment_strategy": package.get('deployment', {}).get('kubernetes', {}).get('ingress', {}) and "kubernetes" or "docker"
                },
                "mcp_servers_used": ["architecture-design", "code-generation", "testing-automation", "docker-management", "kubernetes-cluster"],
                "success": True,
                "learning_insight": "Complete application architecture designed and packaged"
            })
        except Exception as e:
            logger.debug(f"App creation learning storage failed: {e}")


# Export PAI command
create_app_command = CreateAppPAICommand()


async def execute_pai_create_app(description: str = "", name: str = "",
                                user_requirements: str = "", app_type: str = "",
                                expected_users: str = "", special_features: str = "") -> Dict[str, Any]:
    """
    PAI Create App Command Interface
    Agentic coding: Generate complete applications from natural language specifications
    Parameters match PAI pattern specifications
    """
    inputs = {
        'description': description,
        'name': name,
        'user_requirements': user_requirements,
        'app_type': app_type,
        'expected_users': expected_users,
        'special_features': special_features
    }

    # Initialize command if needed
    command = CreateAppPAICommand()
    return await command.execute(inputs)


# Register PAI command in pattern registry
from utils.patterns import command_registry
from utils.patterns import PAIPattern

create_app_pattern = PAIPattern(
    name='create-app',
    category='development',
    system_prompt='''
    You are an expert full-stack application architect capable of designing and generating complete software applications from natural language requirements. Use architectural design, testing, and deployment MCP servers to create production-ready application packages.''',
    user_template='''Create a complete application based on this specification:

Description: ${description}
Name: ${name}
User Requirements: ${user_requirements}
Application Type: ${app_type}
Expected Users: ${expected_users}
Special Features: ${special_features}

Generate a complete, deployable application package including:
- System architecture design
- Technology stack selection
- Complete codebase structure
- Database schema and relationships
- API specifications and endpoints
- Frontend component architecture
- Testing strategies and test files
- Docker and Kubernetes configurations
- CI/CD pipeline setup
- Deployment and scaling strategies
- Documentation and setup guides
- Security configurations and best practices
- Performance optimization recommendations
- Development timeline and resource estimates

Provide a comprehensive application development plan that can be immediately executed.''',
    variables={
        'description': {'description': 'Natural language description of the desired application', 'required': True},
        'name': {'description': 'Application name (optional)', 'default': 'AI_Generated_App'},
        'user_requirements': {'description': 'Specific user requirements or constraints', 'default': ''},
        'app_type': {'description': 'Type of application (web/mobile/desktop)', 'default': 'web'},
        'expected_users': {'description': 'Expected number of users/concurrent users', 'default': '100'},
        'special_features': {'description': 'Special features or integrations required', 'default': ''}
    },
    model_preference='hybrid',  # PAI routing for complex application creation
    output_format='complete_application_package'
)

command_registry.patterns['create-app'] = create_app_pattern


# For testing/development
if __name__ == "__main__":
    # Example usage
    import asyncio

    async def test():
        result = await execute_pai_create_app(
            description="a social media platform for developers to share code snippets with real-time collaboration, syntax highlighting, and version control integration",
            name="DevCollab",
            user_requirements="must be scalable to 100k users, real-time features, mobile responsive",
            app_type="web",
            expected_users="1000 concurrent",
            special_features="GitHub integration, syntax highlighting, real-time collaboration"
        )

        print(f"PAI Create App Result: {result['success']}")
        if result['success']:
            print(f"App Name: {result['result']['application']['name']}")
            print(f"Tech Stack: {result['metadata']['tech_stack']}")
            print(f"Estimated Dev Time: {result['metadata']['estimated_development_time']}")
            print(f"MCP servers used: {result['metadata']['mcp_servers_used']}")
            print("--- Application Package Generated ---")

            # Show some key sections
            package = result['result']
            print(f"Architecture Components: {len(package.get('architecture', {}).get('components', []))}")
            print(f"Technology Stack: {package.get('technology_stack', {}).get('framework')}")
            print(f"Deployment Strategy: {result['metadata']['deployment_strategy']}")

            if 'development_plan' in package:
                phases = package['development_plan'].get('phases', [])
                print(f"Development Phases: {len(phases)}")
                if phases:
                    print(f"First Phase: {phases[0]['phase']} ({phases[0]['duration']})")

    # Uncomment to test (would require MCP servers running)
    # asyncio.run(test())
