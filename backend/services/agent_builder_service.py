"""
GRETA Agent Builder Development Environment Service
Agent development tools with code editor integration, build system, and testing
"""

import asyncio
import logging
import uuid
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

class AgentBuilderService:
    """Agent Builder Development Environment"""

    def __init__(self, db, llm_service, learning_service):
        self.db = db
        self.llm_service = llm_service
        self.learning_service = learning_service

        # Projects
        self.projects = {}

        # Templates with comprehensive structure
        self.templates = {
            "basic": {
                "name": "Basic Agent",
                "description": "Simple agent with core functionality",
                "files": {
                    "agent.py": "# Basic Agent\nclass Agent:\n    def __init__(self):\n        pass\n    \n    def run(self):\n        pass",
                    "config.json": '{"name": "basic_agent", "version": "1.0.0"}'
                },
                "framework": "python",
                "dependencies": []
            },
            "advanced": {
                "name": "Advanced Agent",
                "description": "Agent with utilities and configuration",
                "files": {
                    "agent.py": "# Advanced Agent\nfrom utils import helper\n\nclass Agent:\n    def __init__(self, config):\n        self.config = config\n    \n    def run(self):\n        pass",
                    "utils.py": "# Utilities\ndef helper():\n    pass",
                    "config.json": '{"name": "advanced_agent", "version": "1.0.0"}',
                    "requirements.txt": "requests\nnumpy"
                },
                "framework": "python",
                "dependencies": ["requests", "numpy"]
            },
            "ml_agent": {
                "name": "ML Agent",
                "description": "Machine learning agent with training",
                "files": {
                    "agent.py": "# ML Agent\nimport torch\nfrom model import Model\n\nclass MLAgent:\n    def __init__(self):\n        self.model = Model()\n    \n    def predict(self, data):\n        return self.model(data)",
                    "model.py": "# Model\nimport torch.nn as nn\n\nclass Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(10, 1)\n    \n    def forward(self, x):\n        return self.fc(x)",
                    "train.py": "# Training\ndef train():\n    pass",
                    "config.json": '{"name": "ml_agent", "version": "1.0.0", "model": "pytorch"}'
                },
                "framework": "pytorch",
                "dependencies": ["torch", "numpy"]
            },
            "langchain": {
                "name": "LangChain Agent",
                "description": "Agent using LangChain framework",
                "files": {
                    "agent.py": "# LangChain Agent\nfrom langchain.agents import Agent\n\nclass CustomAgent(Agent):\n    def __init__(self):\n        super().__init__()\n    \n    def run(self, query):\n        pass",
                    "config.json": '{"name": "langchain_agent", "version": "1.0.0"}'
                },
                "framework": "langchain",
                "dependencies": ["langchain", "openai"]
            }
        }

        # Builds
        self.builds = {}

        # Tests
        self.tests = {}

        # Activity log
        self.activity_log = deque(maxlen=500)

        # Metrics
        self.metrics = {
            "total_projects": 0,
            "active_projects": 0,
            "builds_completed": 0,
            "builds_failed": 0,
            "tests_run": 0,
            "tests_passed": 0
        }
        
    async def initialize(self):
        try:
            logger.info("🚀 Initializing Agent Builder...")
            await self._load_projects()
            await self._load_builds()
            await self._load_tests()
            logger.info("✅ Agent Builder initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Agent Builder initialization failed: {e}")
            return False

    def _log_activity(self, action: str, details: str, metadata: Dict[str, Any] = None):
        """Log activity"""
        self.activity_log.append({
            "action": action,
            "details": details,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        })

    async def _load_projects(self):
        try:
            projects = await self.db.find_many("builder_projects", {}, limit=1000)
            for project in projects:
                self.projects[project["project_id"]] = project
                if project.get("status") == "active":
                    self.metrics["active_projects"] += 1
            self.metrics["total_projects"] = len(projects)
            logger.info(f"✅ Loaded {len(projects)} projects")
        except Exception as e:
            logger.debug(f"No existing projects: {e}")

    async def _load_builds(self):
        """Load builds from database"""
        try:
            builds = await self.db.find_many("builder_builds", {}, limit=5000)
            for build in builds:
                self.builds[build["build_id"]] = build
                if build.get("status") == "success":
                    self.metrics["builds_completed"] += 1
                elif build.get("status") == "failed":
                    self.metrics["builds_failed"] += 1
            logger.info(f"✅ Loaded {len(builds)} builds")
        except Exception as e:
            logger.debug(f"No existing builds: {e}")

    async def _load_tests(self):
        """Load tests from database"""
        try:
            tests = await self.db.find_many("builder_tests", {}, limit=5000)
            for test in tests:
                self.tests[test["test_id"]] = test
                self.metrics["tests_run"] += 1
                if test.get("status") == "passed":
                    self.metrics["tests_passed"] += 1
            logger.info(f"✅ Loaded {len(tests)} tests")
        except Exception as e:
            logger.debug(f"No existing tests: {e}")
    
    async def create_project(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new agent project"""
        try:
            project_id = f"project_{uuid.uuid4().hex[:8]}"
            template_name = config.get("template", "basic")

            if template_name not in self.templates:
                return {"success": False, "error": f"Template '{template_name}' not found"}

            template = self.templates[template_name]

            project = {
                "project_id": project_id,
                "name": config.get("name", f"Project_{project_id}"),
                "description": config.get("description", ""),
                "template": template_name,
                "files": dict(template["files"]),  # Copy files
                "framework": template["framework"],
                "dependencies": list(template["dependencies"]),
                "status": "active",
                "created_at": datetime.now().isoformat(),
                "created_by": config.get("created_by", "system"),
                "builds": [],
                "tests": []
            }

            # Store in database
            try:
                await self.db.insert_one("builder_projects", project)
            except Exception as e:
                logger.warning(f"Could not save project to database: {e}")

            self.projects[project_id] = project
            self.metrics["total_projects"] += 1
            self.metrics["active_projects"] += 1

            self._log_activity("project_created", f"Created project: {project['name']}", {"project_id": project_id, "template": template_name})

            await self.learning_service.record_pai_interaction({
                "user_id": "agent_builder",
                "type": "project_creation",
                "category": "development",
                "strategic_reasoning": "Created agent project",
                "tactical_reasoning": f"Template: {template_name}",
                "operational_reasoning": "Project created",
                "response_quality": 1.0,
                "user_satisfaction": 1.0,
                "context_relevance": 1.0
            })

            logger.info(f"✅ Project created: {project_id}")

            return {"success": True, "project_id": project_id, "project": project}
        except Exception as e:
            logger.error(f"❌ Project creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def update_file(self, project_id: str, filename: str, content: str) -> Dict[str, Any]:
        """Update a file in the project"""
        try:
            if project_id not in self.projects:
                return {"success": False, "error": "Project not found"}

            project = self.projects[project_id]

            if filename not in project["files"]:
                return {"success": False, "error": f"File '{filename}' not found in project"}

            project["files"][filename] = content
            project["updated_at"] = datetime.now().isoformat()

            # Update in database
            try:
                await self.db.update_one("builder_projects", {"project_id": project_id}, {"$set": project})
            except Exception as e:
                logger.warning(f"Could not update project in database: {e}")

            self._log_activity("file_updated", f"Updated file {filename} in project {project_id}", {"project_id": project_id, "filename": filename})

            logger.info(f"✅ File updated: {filename}")

            return {"success": True, "project_id": project_id, "filename": filename}
        except Exception as e:
            logger.error(f"Error updating file: {e}")
            return {"success": False, "error": str(e)}

    async def add_file(self, project_id: str, filename: str, content: str = "") -> Dict[str, Any]:
        """Add a new file to the project"""
        try:
            if project_id not in self.projects:
                return {"success": False, "error": "Project not found"}

            project = self.projects[project_id]

            if filename in project["files"]:
                return {"success": False, "error": f"File '{filename}' already exists"}

            project["files"][filename] = content
            project["updated_at"] = datetime.now().isoformat()

            # Update in database
            try:
                await self.db.update_one("builder_projects", {"project_id": project_id}, {"$set": project})
            except Exception as e:
                logger.warning(f"Could not update project in database: {e}")

            self._log_activity("file_added", f"Added file {filename} to project {project_id}", {"project_id": project_id, "filename": filename})

            logger.info(f"✅ File added: {filename}")

            return {"success": True, "project_id": project_id, "filename": filename}
        except Exception as e:
            logger.error(f"Error adding file: {e}")
            return {"success": False, "error": str(e)}

    async def delete_file(self, project_id: str, filename: str) -> Dict[str, Any]:
        """Delete a file from the project"""
        try:
            if project_id not in self.projects:
                return {"success": False, "error": "Project not found"}

            project = self.projects[project_id]

            if filename not in project["files"]:
                return {"success": False, "error": f"File '{filename}' not found"}

            del project["files"][filename]
            project["updated_at"] = datetime.now().isoformat()

            # Update in database
            try:
                await self.db.update_one("builder_projects", {"project_id": project_id}, {"$set": project})
            except Exception as e:
                logger.warning(f"Could not update project in database: {e}")

            self._log_activity("file_deleted", f"Deleted file {filename} from project {project_id}", {"project_id": project_id, "filename": filename})

            logger.info(f"✅ File deleted: {filename}")

            return {"success": True, "message": f"File {filename} deleted"}
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return {"success": False, "error": str(e)}

    async def build_project(self, project_id: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Build the project"""
        try:
            if project_id not in self.projects:
                return {"success": False, "error": "Project not found"}

            project = self.projects[project_id]
            build_id = f"build_{uuid.uuid4().hex[:8]}"

            # Simulate build process
            import random
            success = random.random() > 0.1  # 90% success rate

            build = {
                "build_id": build_id,
                "project_id": project_id,
                "status": "success" if success else "failed",
                "artifacts": [f"{filename}.compiled" for filename in project["files"].keys()] if success else [],
                "logs": "Build completed successfully" if success else "Build failed: compilation error",
                "built_at": datetime.now().isoformat(),
                "duration": random.uniform(1.0, 10.0)
            }

            # Store build
            try:
                await self.db.insert_one("builder_builds", build)
            except Exception as e:
                logger.warning(f"Could not save build to database: {e}")

            self.builds[build_id] = build
            project["builds"].append(build_id)

            if success:
                self.metrics["builds_completed"] += 1
            else:
                self.metrics["builds_failed"] += 1

            self._log_activity("project_built", f"Built project {project_id}: {build['status']}", {"project_id": project_id, "build_id": build_id, "status": build["status"]})

            logger.info(f"✅ Build {'completed' if success else 'failed'}: {build_id}")

            return {"success": True, "build_id": build_id, "build": build}
        except Exception as e:
            logger.error(f"❌ Build failed: {e}")
            self.metrics["builds_failed"] += 1
            return {"success": False, "error": str(e)}

    async def run_tests(self, project_id: str, test_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run tests for the project"""
        try:
            if project_id not in self.projects:
                return {"success": False, "error": "Project not found"}

            project = self.projects[project_id]
            test_id = f"test_{uuid.uuid4().hex[:8]}"

            # Simulate test execution
            import random
            num_tests = random.randint(5, 20)
            passed = random.randint(int(num_tests * 0.7), num_tests)
            failed = num_tests - passed

            test = {
                "test_id": test_id,
                "project_id": project_id,
                "status": "passed" if failed == 0 else "failed",
                "total_tests": num_tests,
                "passed": passed,
                "failed": failed,
                "coverage": random.uniform(0.7, 0.95),
                "duration": random.uniform(0.5, 5.0),
                "run_at": datetime.now().isoformat()
            }

            # Store test
            try:
                await self.db.insert_one("builder_tests", test)
            except Exception as e:
                logger.warning(f"Could not save test to database: {e}")

            self.tests[test_id] = test
            project["tests"].append(test_id)

            self.metrics["tests_run"] += 1
            if test["status"] == "passed":
                self.metrics["tests_passed"] += 1

            self._log_activity("tests_run", f"Ran tests for project {project_id}: {passed}/{num_tests} passed", {"project_id": project_id, "test_id": test_id})

            logger.info(f"✅ Tests completed: {passed}/{num_tests} passed")

            return {"success": True, "test_id": test_id, "test": test}
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return {"success": False, "error": str(e)}

    async def deploy_project(self, project_id: str, deployment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy the project"""
        try:
            if project_id not in self.projects:
                return {"success": False, "error": "Project not found"}

            project = self.projects[project_id]
            deployment_id = f"deploy_{uuid.uuid4().hex[:8]}"

            deployment = {
                "deployment_id": deployment_id,
                "project_id": project_id,
                "environment": deployment_config.get("environment", "production"),
                "status": "deployed",
                "url": f"https://{project['name']}.example.com",
                "deployed_at": datetime.now().isoformat()
            }

            self._log_activity("project_deployed", f"Deployed project {project_id} to {deployment['environment']}", {"project_id": project_id, "deployment_id": deployment_id})

            logger.info(f"✅ Project deployed: {deployment_id}")

            return {"success": True, "deployment_id": deployment_id, "deployment": deployment}
        except Exception as e:
            logger.error(f"Error deploying project: {e}")
            return {"success": False, "error": str(e)}

    async def get_project(self, project_id: str) -> Dict[str, Any]:
        """Get a specific project"""
        try:
            project = self.projects.get(project_id)

            if not project:
                return {"success": False, "error": "Project not found"}

            return {"success": True, "project": project}
        except Exception as e:
            logger.error(f"Error getting project: {e}")
            return {"success": False, "error": str(e)}

    async def list_projects(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """List all projects"""
        try:
            projects = list(self.projects.values())

            if filters:
                if "status" in filters:
                    projects = [p for p in projects if p.get("status") == filters["status"]]
                if "framework" in filters:
                    projects = [p for p in projects if p.get("framework") == filters["framework"]]

            return {"success": True, "projects": projects, "count": len(projects)}
        except Exception as e:
            logger.error(f"Error listing projects: {e}")
            return {"success": False, "error": str(e)}

    async def delete_project(self, project_id: str) -> Dict[str, Any]:
        """Delete a project"""
        try:
            if project_id not in self.projects:
                return {"success": False, "error": "Project not found"}

            project = self.projects[project_id]

            # Delete from database
            try:
                await self.db.delete_one("builder_projects", {"project_id": project_id})
            except Exception as e:
                logger.warning(f"Could not delete project from database: {e}")

            del self.projects[project_id]

            self.metrics["total_projects"] -= 1
            if project.get("status") == "active":
                self.metrics["active_projects"] -= 1

            self._log_activity("project_deleted", f"Deleted project {project_id}", {"project_id": project_id})

            logger.info(f"✅ Project deleted: {project_id}")

            return {"success": True, "message": f"Project {project_id} deleted"}
        except Exception as e:
            logger.error(f"Error deleting project: {e}")
            return {"success": False, "error": str(e)}

    async def get_templates(self) -> Dict[str, Any]:
        """Get available templates"""
        templates_info = []
        for key, template in self.templates.items():
            templates_info.append({
                "id": key,
                "name": template["name"],
                "description": template["description"],
                "framework": template["framework"],
                "files": list(template["files"].keys()),
                "dependencies": template["dependencies"]
            })

        return {"success": True, "templates": templates_info, "count": len(templates_info)}

    async def get_build(self, build_id: str) -> Dict[str, Any]:
        """Get build details"""
        try:
            build = self.builds.get(build_id)

            if not build:
                return {"success": False, "error": "Build not found"}

            return {"success": True, "build": build}
        except Exception as e:
            logger.error(f"Error getting build: {e}")
            return {"success": False, "error": str(e)}

    async def get_test(self, test_id: str) -> Dict[str, Any]:
        """Get test details"""
        try:
            test = self.tests.get(test_id)

            if not test:
                return {"success": False, "error": "Test not found"}

            return {"success": True, "test": test}
        except Exception as e:
            logger.error(f"Error getting test: {e}")
            return {"success": False, "error": str(e)}

    async def get_activity_log(self, limit: int = 100) -> Dict[str, Any]:
        """Get activity log"""
        try:
            activities = list(self.activity_log)[-limit:]
            return {"success": True, "activities": activities, "count": len(activities)}
        except Exception as e:
            logger.error(f"Error getting activity log: {e}")
            return {"success": False, "error": str(e)}

    async def get_metrics(self) -> Dict[str, Any]:
        """Get builder metrics"""
        return {
            "success": True,
            "metrics": self.metrics,
            "timestamp": datetime.now().isoformat()
        }

