"""
GRETA MCP Server Manager
Easy MCP server addition, validation, and management - as requested by user
CLI tool for adding MCP servers to the ecosystem without manual JSON editing
"""
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import httpx
import subprocess
from datetime import datetime

logger = logging.getLogger(__name__)


class MCPServerManager:
    """
    MCP Server Manager for Greta PAI
    Allows easy addition and management of MCP servers
    """

    def __init__(self, config_path: str = "enhanced_mcp_config.json"):
        self.config_path = Path(config_path)
        self.available_mcp_servers = self._load_available_mcp_servers()

    def _load_available_mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """Load list of available MCP servers that can be added"""
        return {
            # Research & Content
            "exa-search": {
                "description": "Advanced semantic web search with AI understanding",
                "type": "http",
                "command": "npx @modelcontextprotocol/exa-search",
                "url": "https://api.exa.ai/search",
                "required_env": ["EXA_API_KEY"],
                "category": "research"
            },
            "context7": {
                "description": "Advanced conversational context management",
                "type": "http",
                "command": "npx @modelcontextprotocol/context7",
                "url": "https://api.context7.ai/v1",
                "required_env": ["CONTEXT7_API_KEY", "CONTEXT7_PROJECT_ID"],
                "category": "conversation"
            },
            "magic-ui": {
                "description": "AI-powered UI component generation",
                "type": "http",
                "command": "npx @modelcontextprotocol/magic-ui",
                "url": "https://api.magic.design/v1",
                "required_env": ["MAGIC_UI_API_KEY"],
                "category": "design"
            },

            # Development & Coding
            "github-advanced": {
                "description": "Advanced GitHub operations with insights",
                "type": "stdlib",
                "command": "npx @modelcontextprotocol/github",
                "required_env": ["GITHUB_TOKEN"],
                "category": "development"
            },
            "code-analysis-advanced": {
                "description": "Advanced semantic code analysis",
                "type": "local",
                "command": "npx @modelcontextprotocol/code-analysis",
                "required_env": ["CODE_ANALYSIS_API_KEY"],
                "category": "development"
            },
            "testing-automation": {
                "description": "Automated test generation and execution",
                "type": "local",
                "command": "npx @modelcontextprotocol/testing-automation",
                "required_env": ["TESTING_API_KEY"],
                "category": "development"
            },

            # Deployment & Infrastructure
            "docker-advanced": {
                "description": "Advanced container management and orchestration",
                "type": "local",
                "command": "npx @modelcontextprotocol/docker-management",
                "required_env": ["DOCKER_API_KEY"],
                "category": "infrastructure"
            },
            "kubernetes-advanced": {
                "description": "Kubernetes cluster management and deployment",
                "type": "local",
                "command": "npx @modelcontextprotocol/kubernetes-cluster",
                "required_env": ["KUBECONFIG", "KUBERNETES_API_KEY"],
                "category": "infrastructure"
            },

            # Learning & Analytics
            "learning-analytics": {
                "description": "AI learning pattern analysis and insights",
                "type": "local",
                "command": "npx @modelcontextprotocol/learning-analytics",
                "required_env": ["LEARNING_ANALYTICS_API"],
                "category": "analytics"
            },
            "performance-monitoring": {
                "description": "System performance monitoring and optimization",
                "type": "local",
                "command": "npx @modelcontextprotocol/performance-monitoring",
                "required_env": ["PERFORMANCE_API_KEY"],
                "category": "analytics"
            },

            # Communication & Productivity
            "slack-advanced": {
                "description": "Advanced Slack integration and automation",
                "type": "http",
                "command": "npx @modelcontextprotocol/slack",
                "url": "https://slack.com/api/",
                "required_env": ["SLACK_BOT_TOKEN", "SLACK_TEAM_ID"],
                "category": "communication"
            },
            "email-advanced": {
                "description": "Advanced email processing and analysis",
                "type": "http",
                "command": "npx @modelcontextprotocol/email",
                "required_env": ["EMAIL_PROVIDER", "EMAIL_CREDENTIALS"],
                "category": "communication"
            },

            # Finance & Business
            "stripe-advanced": {
                "description": "Advanced payment processing and financial operations",
                "type": "http",
                "command": "bunx @stripe/mcp --tools=all",
                "url": "https://api.stripe.com/v1",
                "required_env": ["STRIPE_SECRET_KEY"],
                "category": "business"
            },
            "financial-analysis": {
                "description": "Advanced financial analysis and planning",
                "type": "http",
                "command": "npx @modelcontextprotocol/financial-analysis",
                "required_env": ["FINANCIAL_API_KEY"],
                "category": "business"
            },

            # Content & Media
            "content-management": {
                "description": "Content management system integration",
                "type": "http",
                "command": "npx @modelcontextprotocol/content-management",
                "required_env": ["CONTENT_API_KEY"],
                "category": "content"
            },
            "documentation-generator": {
                "description": "Automated documentation generation",
                "type": "local",
                "command": "npx @modelcontextprotocol/documentation-generator",
                "required_env": ["DOCS_API_KEY"],
                "category": "content"
            },

            # Security & Compliance
            "security-scanner-advanced": {
                "description": "Advanced security vulnerability scanning",
                "type": "local",
                "command": "npx @modelcontextprotocol/security-scanner",
                "required_env": ["SECURITY_API_KEY"],
                "category": "security"
            },
            "dependency-analysis": {
                "description": "Package and dependency security analysis",
                "type": "local",
                "command": "npx @modelcontextprotocol/dependency-analysis",
                "required_env": ["DEPENDENCY_API"],
                "category": "security"
            },

            # Personal & Lifestyle
            "weather-advanced": {
                "description": "Advanced weather data and forecasting",
                "type": "http",
                "command": "npx @modelcontextprotocol/weather",
                "url": "https://api.weatherapi.com/v1",
                "required_env": ["WEATHER_API_KEY"],
                "category": "personal"
            },
            "calendar-advanced": {
                "description": "Advanced calendar management and scheduling",
                "type": "http",
                "command": "npx @modelcontextprotocol/calendar",
                "required_env": ["CALENDAR_API_KEY"],
                "category": "personal"
            },
            "health-monitoring": {
                "description": "Personal health monitoring and analysis",
                "type": "http",
                "command": "npx @modelcontextprotocol/health-monitoring",
                "required_env": ["HEALTH_API_KEY"],
                "category": "personal"
            },

            # Knowledge & Research
            "arxiv-advanced": {
                "description": "Advanced academic paper search and analysis",
                "type": "http",
                "command": "npx @modelcontextprotocol/arxiv",
                "url": "https://export.arxiv.org/api/query",
                "category": "research"
            },
            "patent-search": {
                "description": "Patent database search and analysis",
                "type": "http",
                "command": "npx @modelcontextprotocol/patent-search",
                "required_env": ["PATENT_API_KEY"],
                "category": "research"
            },

            # Meta & Management
            "mcp-compass": {
                "description": "MCP server meta-management and optimization",
                "type": "local",
                "command": "npx @modelcontextprotocol/compass",
                "required_env": ["COMPASS_API_KEY"],
                "category": "management"
            },
            "self-improve": {
                "description": "AI system performance analysis and improvement",
                "type": "local",
                "command": "npx @modelcontextprotocol/self-improve",
                "required_env": ["SELF_IMPROVE_API", "GITHUB_OAUTH"],
                "category": "management"
            },

            # Specialized Domains
            "medical-analysis": {
                "description": "Medical data analysis and research",
                "type": "http",
                "command": "npx @modelcontextprotocol/medical-analysis",
                "required_env": ["MEDICAL_API_KEY"],
                "category": "specialized"
            },
            "legal-research": {
                "description": "Legal document analysis and research",
                "type": "http",
                "command": "npx @modelcontextprotocol/legal-research",
                "required_env": ["LEGAL_API_KEY"],
                "category": "specialized"
            },
            "market-research": {
                "description": "Advanced market research and competitive analysis",
                "type": "http",
                "command": "npx @modelcontextprotocol/market-research",
                "required_env": ["MARKET_API_KEY"],
                "category": "specialized"
            }
        }

    def list_available_servers(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available MCP servers, optionally filtered by category"""
        servers = []
        for name, config in self.available_mcp_servers.items():
            if category is None or config.get('category') == category:
                servers.append({
                    'name': name,
                    'description': config['description'],
                    'category': config['category'],
                    'type': config['type'],
                    'required_env': config.get('required_env', [])
                })
        return servers

    def get_server_details(self, server_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific MCP server"""
        return self.available_mcp_servers.get(server_name)

    async def add_server_interactive(self, server_name: str) -> bool:
        """Interactively add an MCP server with user input for configuration"""
        if server_name not in self.available_mcp_servers:
            logger.error(f"Server '{server_name}' not found in available servers")
            return False

        server_config = self.available_mcp_servers[server_name]

        print(f"\n🛠️  Adding MCP Server: {server_name}")
        print(f"📝 Description: {server_config['description']}")
        print(f"🏷️  Category: {server_config['category']}")
        print(f"🔧 Type: {server_config['type']}")

        # Check for required environment variables
        required_env = server_config.get('required_env', [])
        if required_env:
            print(f"\n🔑 Required Environment Variables: {', '.join(required_env)}")
            env_values = {}

            for env_var in required_env:
                current_value = input(f"Enter value for {env_var} (or press Enter to skip): ").strip()
                if current_value:
                    env_values[env_var] = current_value

            # Validate environment setup
            if not await self._validate_environment_config(env_values):
                print("❌ Environment configuration validation failed")
                return False

            server_config['env'] = env_values

        # Test server connection
        if not await self._test_server_connection(server_config):
            print("⚠️  Server connection test failed, but continuing with addition")
            proceed = input("Continue adding server? (y/N): ").lower().strip()
            if proceed != 'y':
                return False

        # Add to configuration
        success = await self.add_server_to_config(server_name, server_config)
        if success:
            print(f"✅ Successfully added {server_name} to MCP configuration!")

            # Offer to restart MCP services
            restart = input("Restart MCP services now? (y/N): ").lower().strip()
            if restart == 'y':
                await self.restart_mcp_services()

        return success

    async def add_server_to_config(self, server_name: str, server_config: Dict[str, Any]) -> bool:
        """Add MCP server to the enhanced configuration"""
        try:
            # Load current configuration
            config_data = await self._load_current_config()

            # Add server
            config_data['mcpServers'][server_name] = server_config

            # Save configuration
            success = await self._save_config(config_data)
            if success:
                logger.info(f"Added MCP server {server_name} to configuration")

                # Update MCP Compass if it's available
                if 'mcp-compass' in config_data['mcpServers']:
                    await self._update_mcp_compass(server_name, server_config)

            return success

        except Exception as e:
            logger.error(f"Failed to add server to config: {e}")
            return False

    async def remove_server(self, server_name: str) -> bool:
        """Remove an MCP server from configuration"""
        try:
            config_data = await self._load_current_config()

            if server_name in config_data['mcpServers']:
                del config_data['mcpServers'][server_name]
                await self._save_config(config_data)

                # Update MCP Compass
                await self._remove_from_mcp_compass(server_name)

                print(f"✅ Removed {server_name} from MCP configuration")
                return True
            else:
                print(f"❌ Server {server_name} not found in configuration")
                return False

        except Exception as e:
            logger.error(f"Failed to remove server: {e}")
            return False

    async def list_installed_servers(self) -> List[str]:
        """List currently installed MCP servers"""
        try:
            config_data = await self._load_current_config()
            return list(config_data.get('mcpServers', {}).keys())
        except Exception:
            return []

    async def validate_server_config(self, server_name: str) -> Dict[str, Any]:
        """Validate MCP server configuration and connectivity"""
        config_data = await self._load_current_config()
        server_config = config_data.get('mcpServers', {}).get(server_name)

        if not server_config:
            return {'valid': False, 'error': 'Server not found in configuration'}

        # Check required environment variables
        required_env = server_config.get('env', {})
        missing_env = []
        for env_var in required_env.keys():
            if not self._check_env_variable(env_var):
                missing_env.append(env_var)

        if missing_env:
            return {
                'valid': False,
                'error': f'Missing environment variables: {", ".join(missing_env)}'
            }

        # Test server connectivity
        connection_test = await self._test_server_connection(server_config)

        return {
            'valid': connection_test,
            'server_name': server_name,
            'type': server_config.get('type'),
            'category': server_config.get('description', 'Unknown'),
            'required_env_present': len(missing_env) == 0,
            'connection_test': connection_test
        }

    async def restart_mcp_services(self) -> bool:
        """Restart MCP services to load new configurations"""
        try:
            print("🔄 Restarting MCP services...")

            # This would depend on how MCP services are managed
            # For example, if using PM2, systemd, or Docker
            result = subprocess.run(['systemctl', 'restart', 'mcp-service'],
                                  capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ MCP services restarted successfully")
                return True
            else:
                print(f"⚠️  MCP service restart failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Failed to restart MCP services: {e}")
            return False

    async def update_server_config(self, server_name: str, updates: Dict[str, Any]) -> bool:
        """Update MCP server configuration"""
        try:
            config_data = await self._load_current_config()

            if server_name not in config_data.get('mcpServers', {}):
                print(f"❌ Server {server_name} not found")
                return False

            config_data['mcpServers'][server_name].update(updates)
            return await self._save_config(config_data)

        except Exception as e:
            logger.error(f"Failed to update server config: {e}")
            return False

    async def _load_current_config(self) -> Dict[str, Any]:
        """Load current MCP configuration"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load config: {e}")

        # Return default structure
        return {'mcpServers': {}}

    async def _save_config(self, config_data: Dict[str, Any]) -> bool:
        """Save MCP configuration"""
        try:
            # Create backup
            backup_path = self.config_path.with_suffix('.backup')
            if self.config_path.exists():
                import shutil
                shutil.copy2(self.config_path, backup_path)

            # Save new config
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)

            print(f"💾 Configuration saved to {self.config_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False

    def _check_env_variable(self, env_var: str) -> bool:
        """Check if environment variable is set and not empty"""
        import os
        value = os.getenv(env_var)
        return value is not None and value.strip() != ''

    async def _validate_environment_config(self, env_values: Dict[str, str]) -> bool:
        """Validate environment configuration"""
        # Check for required environment variables
        missing_vars = []
        for var_name, var_value in env_values.items():
            if not var_value or var_value.strip() == '':
                missing_vars.append(var_name)
            # Additional validation can be added here (e.g., API key format)

        if missing_vars:
            print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
            return False

        print("✅ Environment configuration validated")
        return True

    async def _test_server_connection(self, server_config: Dict[str, Any]) -> bool:
        """Test MCP server connection"""
        server_type = server_config.get('type')
        url = server_config.get('url')

        if server_type == 'http' and url:
            try:
                timeout = httpx.Timeout(10.0)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.get(url, headers={'User-Agent': 'Greta-MCP-Manager/1.0'})
                    return response.status_code < 400
            except Exception as e:
                logger.debug(f"Connection test failed for {url}: {e}")
                return False

        elif server_type == 'local':
            # Test if command is available
            command = server_config.get('command', '').split()[0]
            try:
                result = subprocess.run(['which', command],
                                      capture_output=True, text=True)
                return result.returncode == 0
            except Exception:
                return False

        # For stdlib types, assume available
        elif server_type == 'stdlib':
            return True

        return False

    async def _update_mcp_compass(self, server_name: str, server_config: Dict[str, Any]) -> None:
        """Update MCP Compass with new server information"""
        try:
            # This would integrate with MCP Compass to register the new server
            # For now, just log the action
            logger.info(f"Would update MCP Compass with server: {server_name}")
        except Exception as e:
            logger.debug(f"MCP Compass update failed: {e}")

    async def _remove_from_mcp_compass(self, server_name: str) -> None:
        """Remove server from MCP Compass"""
        try:
            logger.info(f"Would remove {server_name} from MCP Compass")
        except Exception as e:
            logger.debug(f"MCP Compass removal failed: {e}")


# CLI Interface
def main():
    """CLI interface for MCP Server Manager"""
    import argparse

    parser = argparse.ArgumentParser(description='Greta MCP Server Manager')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # List command
    list_parser = subparsers.add_parser('list', help='List MCP servers')
    list_parser.add_argument('--category', help='Filter by category')
    list_parser.add_argument('--installed', action='store_true', help='List installed servers')

    # Add command
    add_parser = subparsers.add_parser('add', help='Add MCP server')
    add_parser.add_argument('name', help='Server name to add')

    # Remove command
    remove_parser = subparsers.add_parser('remove', help='Remove MCP server')
    remove_parser.add_argument('name', help='Server name to remove')

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate MCP server')
    validate_parser.add_argument('name', help='Server name to validate')

    # Restart command
    subparsers.add_parser('restart', help='Restart MCP services')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize manager
    manager = MCPServerManager()

    if args.command == 'list':
        if args.installed:
            installed = asyncio.run(manager.list_installed_servers())
            if installed:
                print("📦 Installed MCP Servers:")
                for server in installed:
                    print(f"  ✓ {server}")
            else:
                print("📦 No MCP servers installed")
        else:
            servers = manager.list_available_servers(args.category)
            if args.category:
                print(f"🔍 Available {args.category} MCP Servers:")
            else:
                print("🔍 All Available MCP Servers:")

            for server in servers:
                print(f"  {server['name']} - {server['description']} [{server['category']}]")

    elif args.command == 'add':
        success = asyncio.run(manager.add_server_interactive(args.name))
        exit(0 if success else 1)

    elif args.command == 'remove':
        success = asyncio.run(manager.remove_server(args.name))
        exit(0 if success else 1)

    elif args.command == 'validate':
        result = asyncio.run(manager.validate_server_config(args.name))
        print(f"🔍 Validation Results for {args.name}:")
        for key, value in result.items():
            status = "✅" if value in [True, "Present", "Connected"] else "❌" if value in [False, "Missing", "Not connected"] else "ℹ️"
            print(f"  {status} {key}: {value}")

    elif args.command == 'restart':
        success = asyncio.run(manager.restart_mcp_services())
        exit(0 if success else 1)


if __name__ == '__main__':
    main()
