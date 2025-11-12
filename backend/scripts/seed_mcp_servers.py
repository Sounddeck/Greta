#!/usr/bin/env python3
"""
Seed Default MCP Servers
Creates example MCP servers in the database for Agent Builder
"""
import asyncio
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database import Database
from models.mcp_models import MCPServerCreate, DiscoverySource, MCPToolType, MCPToolBase
from services.mcp_service import MCPHubService

async def seed_mcp_servers():
    """Seed default MCP servers into the database"""
    print("🌱 Seeding MCP Servers...")
    print("=" * 60)
    
    # Connect to database
    db = Database()
    await db.connect()
    service = MCPHubService(db)
    
    # Check if servers already exist
    existing_servers = await service.list_servers(limit=1000)
    if len(existing_servers) > 0:
        print(f"\n⚠️  Found {len(existing_servers)} existing MCP servers")
        response = input("Do you want to add more servers anyway? (y/n): ")
        if response.lower() != 'y':
            print("❌ Cancelled")
            return
    
    # Default MCP servers to seed
    default_servers = [
        {
            "name": "Filesystem MCP Server",
            "description": "File system operations - read, write, search files and directories",
            "version": "1.0.0",
            "author": "Anthropic",
            "homepage": "https://github.com/modelcontextprotocol/servers",
            "repository": "https://github.com/modelcontextprotocol/servers",
            "license": "MIT",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/directory"],
            "capabilities": ["file_read", "file_write", "directory_list", "file_search"],
            "tags": ["filesystem", "files", "storage"],
            "discovery_source": DiscoverySource.MANUAL,
            "auto_start": False,
            "tools": [
                {
                    "name": "read_file",
                    "description": "Read the complete contents of a file from the file system",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "path": {"type": "string", "description": "Path to the file to read"}
                    }
                },
                {
                    "name": "write_file",
                    "description": "Create a new file or overwrite an existing file with new content",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "path": {"type": "string", "description": "Path where the file should be written"},
                        "content": {"type": "string", "description": "Content to write to the file"}
                    }
                },
                {
                    "name": "list_directory",
                    "description": "Get a detailed listing of all files and directories in a specified path",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "path": {"type": "string", "description": "Path to the directory to list"}
                    }
                },
                {
                    "name": "search_files",
                    "description": "Recursively search for files and directories matching a pattern",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "path": {"type": "string", "description": "Starting directory path"},
                        "pattern": {"type": "string", "description": "Search pattern (glob format)"}
                    }
                }
            ]
        },
        {
            "name": "GitHub MCP Server",
            "description": "GitHub repository operations - search code, create issues, manage PRs",
            "version": "1.0.0",
            "author": "Anthropic",
            "homepage": "https://github.com/modelcontextprotocol/servers",
            "repository": "https://github.com/modelcontextprotocol/servers",
            "license": "MIT",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "your-github-token"},
            "capabilities": ["code_search", "issue_management", "pr_management", "repository_info"],
            "tags": ["github", "git", "version-control", "code"],
            "discovery_source": DiscoverySource.MANUAL,
            "auto_start": False,
            "tools": [
                {
                    "name": "search_repositories",
                    "description": "Search for GitHub repositories",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "query": {"type": "string", "description": "Search query"}
                    }
                },
                {
                    "name": "create_issue",
                    "description": "Create a new issue in a repository",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "owner": {"type": "string", "description": "Repository owner"},
                        "repo": {"type": "string", "description": "Repository name"},
                        "title": {"type": "string", "description": "Issue title"},
                        "body": {"type": "string", "description": "Issue description"}
                    }
                },
                {
                    "name": "search_code",
                    "description": "Search for code across GitHub repositories",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "query": {"type": "string", "description": "Code search query"}
                    }
                }
            ]
        },
        {
            "name": "Google Maps MCP Server",
            "description": "Google Maps integration - geocoding, directions, places search",
            "version": "1.0.0",
            "author": "Anthropic",
            "homepage": "https://github.com/modelcontextprotocol/servers",
            "repository": "https://github.com/modelcontextprotocol/servers",
            "license": "MIT",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-google-maps"],
            "env": {"GOOGLE_MAPS_API_KEY": "your-google-maps-api-key"},
            "capabilities": ["geocoding", "directions", "places_search", "distance_matrix"],
            "tags": ["maps", "location", "geocoding", "navigation"],
            "discovery_source": DiscoverySource.MANUAL,
            "auto_start": False,
            "tools": [
                {
                    "name": "geocode",
                    "description": "Convert addresses to geographic coordinates",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "address": {"type": "string", "description": "Address to geocode"}
                    }
                },
                {
                    "name": "directions",
                    "description": "Get directions between two locations",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "origin": {"type": "string", "description": "Starting location"},
                        "destination": {"type": "string", "description": "Destination location"}
                    }
                },
                {
                    "name": "search_places",
                    "description": "Search for places near a location",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "query": {"type": "string", "description": "Search query"},
                        "location": {"type": "string", "description": "Center location"}
                    }
                }
            ]
        },
        {
            "name": "Brave Search MCP Server",
            "description": "Brave Search API integration - web search, news, images",
            "version": "1.0.0",
            "author": "Anthropic",
            "homepage": "https://github.com/modelcontextprotocol/servers",
            "repository": "https://github.com/modelcontextprotocol/servers",
            "license": "MIT",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-brave-search"],
            "env": {"BRAVE_API_KEY": "your-brave-api-key"},
            "capabilities": ["web_search", "news_search", "image_search"],
            "tags": ["search", "web", "brave", "research"],
            "discovery_source": DiscoverySource.MANUAL,
            "auto_start": False,
            "tools": [
                {
                    "name": "web_search",
                    "description": "Search the web using Brave Search",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "query": {"type": "string", "description": "Search query"},
                        "count": {"type": "integer", "description": "Number of results (default: 10)"}
                    }
                },
                {
                    "name": "local_search",
                    "description": "Search for local businesses and places",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "query": {"type": "string", "description": "Search query"},
                        "location": {"type": "string", "description": "Location to search near"}
                    }
                }
            ]
        },
        {
            "name": "Slack MCP Server",
            "description": "Slack workspace integration - send messages, search, manage channels",
            "version": "1.1.25",
            "author": "korotovsky",
            "homepage": "https://github.com/korotovsky/slack-mcp-server",
            "repository": "https://github.com/korotovsky/slack-mcp-server",
            "license": "MIT",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "slack-mcp-server"],
            "env": {
                "SLACK_MCP_XOXC_TOKEN": "your-xoxc-token",
                "SLACK_MCP_XOXD_TOKEN": "your-xoxd-token",
                "SLACK_MCP_PORT": "13080"
            },
            "capabilities": ["messaging", "channel_management", "search", "history"],
            "tags": ["slack", "communication", "messaging", "collaboration"],
            "discovery_source": DiscoverySource.MANUAL,
            "auto_start": False,
            "tools": [
                {
                    "name": "conversations_history",
                    "description": "Fetch conversation history from a Slack channel",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "channel": {"type": "string", "description": "Channel ID"}
                    }
                },
                {
                    "name": "conversations_add_message",
                    "description": "Send a message to a Slack channel",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "channel": {"type": "string", "description": "Channel ID"},
                        "text": {"type": "string", "description": "Message text"}
                    }
                },
                {
                    "name": "channels_list",
                    "description": "List all channels in the workspace",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {}
                }
            ]
        },
        {
            "name": "PostgreSQL MCP Server",
            "description": "PostgreSQL database operations - query, insert, update, delete data",
            "version": "1.0.0",
            "author": "Anthropic",
            "homepage": "https://github.com/modelcontextprotocol/servers",
            "repository": "https://github.com/modelcontextprotocol/servers",
            "license": "MIT",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-postgres"],
            "env": {
                "POSTGRES_CONNECTION_STRING": "postgresql://user:password@localhost:5432/database"
            },
            "capabilities": ["database_query", "data_manipulation", "schema_inspection"],
            "tags": ["database", "postgresql", "sql", "data"],
            "discovery_source": DiscoverySource.MANUAL,
            "auto_start": False,
            "tools": [
                {
                    "name": "query",
                    "description": "Execute a SQL query on the PostgreSQL database",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "sql": {"type": "string", "description": "SQL query to execute"}
                    }
                },
                {
                    "name": "list_tables",
                    "description": "List all tables in the database",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {}
                },
                {
                    "name": "describe_table",
                    "description": "Get schema information for a table",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "table": {"type": "string", "description": "Table name"}
                    }
                }
            ]
        },
        {
            "name": "Docker MCP Server",
            "description": "Docker container management - list, start, stop, inspect containers and images",
            "version": "1.0.0",
            "author": "Community",
            "homepage": "https://github.com/modelcontextprotocol/servers",
            "repository": "https://github.com/modelcontextprotocol/servers",
            "license": "MIT",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-docker"],
            "capabilities": ["container_management", "image_management", "network_management"],
            "tags": ["docker", "containers", "devops", "infrastructure"],
            "discovery_source": DiscoverySource.MANUAL,
            "auto_start": False,
            "tools": [
                {
                    "name": "list_containers",
                    "description": "List all Docker containers",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "all": {"type": "boolean", "description": "Show all containers (default: running only)"}
                    }
                },
                {
                    "name": "start_container",
                    "description": "Start a Docker container",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "container_id": {"type": "string", "description": "Container ID or name"}
                    }
                },
                {
                    "name": "stop_container",
                    "description": "Stop a Docker container",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "container_id": {"type": "string", "description": "Container ID or name"}
                    }
                },
                {
                    "name": "list_images",
                    "description": "List all Docker images",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {}
                }
            ]
        },
        {
            "name": "Email SMTP MCP Server",
            "description": "Email operations - send emails, manage templates, track delivery",
            "version": "1.0.0",
            "author": "Community",
            "homepage": "https://github.com/modelcontextprotocol/servers",
            "repository": "https://github.com/modelcontextprotocol/servers",
            "license": "MIT",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-email"],
            "env": {
                "SMTP_HOST": "smtp.gmail.com",
                "SMTP_PORT": "587",
                "SMTP_USER": "your-email@gmail.com",
                "SMTP_PASSWORD": "your-app-password"
            },
            "capabilities": ["email_send", "template_management", "delivery_tracking"],
            "tags": ["email", "smtp", "communication", "messaging"],
            "discovery_source": DiscoverySource.MANUAL,
            "auto_start": False,
            "tools": [
                {
                    "name": "send_email",
                    "description": "Send an email",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "to": {"type": "string", "description": "Recipient email address"},
                        "subject": {"type": "string", "description": "Email subject"},
                        "body": {"type": "string", "description": "Email body (HTML or plain text)"}
                    }
                },
                {
                    "name": "send_bulk_email",
                    "description": "Send bulk emails",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "recipients": {"type": "array", "description": "List of recipient email addresses"},
                        "subject": {"type": "string", "description": "Email subject"},
                        "body": {"type": "string", "description": "Email body"}
                    }
                }
            ]
        },
        {
            "name": "Discord MCP Server",
            "description": "Discord bot integration - send messages, manage channels, read history",
            "version": "1.0.0",
            "author": "Community",
            "homepage": "https://github.com/modelcontextprotocol/servers",
            "repository": "https://github.com/modelcontextprotocol/servers",
            "license": "MIT",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-discord"],
            "env": {
                "DISCORD_BOT_TOKEN": "your-bot-token"
            },
            "capabilities": ["messaging", "channel_management", "server_management"],
            "tags": ["discord", "communication", "messaging", "bot"],
            "discovery_source": DiscoverySource.MANUAL,
            "auto_start": False,
            "tools": [
                {
                    "name": "send_message",
                    "description": "Send a message to a Discord channel",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "channel_id": {"type": "string", "description": "Channel ID"},
                        "content": {"type": "string", "description": "Message content"}
                    }
                },
                {
                    "name": "list_channels",
                    "description": "List all channels in a server",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "server_id": {"type": "string", "description": "Server ID"}
                    }
                }
            ]
        },
        {
            "name": "REST API MCP Server",
            "description": "Generic REST API client - make HTTP requests to any API",
            "version": "1.0.0",
            "author": "Community",
            "homepage": "https://github.com/modelcontextprotocol/servers",
            "repository": "https://github.com/modelcontextprotocol/servers",
            "license": "MIT",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-rest-api"],
            "capabilities": ["http_requests", "api_integration", "data_fetching"],
            "tags": ["api", "rest", "http", "integration"],
            "discovery_source": DiscoverySource.MANUAL,
            "auto_start": False,
            "tools": [
                {
                    "name": "get_request",
                    "description": "Make a GET request to an API",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "url": {"type": "string", "description": "API endpoint URL"},
                        "headers": {"type": "object", "description": "Request headers"}
                    }
                },
                {
                    "name": "post_request",
                    "description": "Make a POST request to an API",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "url": {"type": "string", "description": "API endpoint URL"},
                        "body": {"type": "object", "description": "Request body"},
                        "headers": {"type": "object", "description": "Request headers"}
                    }
                }
            ]
        },
        {
            "name": "Google Calendar MCP Server",
            "description": "Google Calendar integration - create events, list calendars, manage schedules",
            "version": "1.0.0",
            "author": "Community",
            "homepage": "https://github.com/modelcontextprotocol/servers",
            "repository": "https://github.com/modelcontextprotocol/servers",
            "license": "MIT",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-google-calendar"],
            "env": {
                "GOOGLE_CALENDAR_API_KEY": "your-api-key",
                "GOOGLE_CALENDAR_CLIENT_ID": "your-client-id",
                "GOOGLE_CALENDAR_CLIENT_SECRET": "your-client-secret"
            },
            "capabilities": ["calendar_management", "event_creation", "schedule_viewing"],
            "tags": ["calendar", "google", "scheduling", "productivity"],
            "discovery_source": DiscoverySource.MANUAL,
            "auto_start": False,
            "tools": [
                {
                    "name": "create_event",
                    "description": "Create a new calendar event",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "summary": {"type": "string", "description": "Event title"},
                        "start_time": {"type": "string", "description": "Start time (ISO 8601)"},
                        "end_time": {"type": "string", "description": "End time (ISO 8601)"}
                    }
                },
                {
                    "name": "list_events",
                    "description": "List upcoming calendar events",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "max_results": {"type": "integer", "description": "Maximum number of events"}
                    }
                }
            ]
        },
        {
            "name": "Jira MCP Server",
            "description": "Jira integration - create issues, update tasks, search projects",
            "version": "1.0.0",
            "author": "Community",
            "homepage": "https://github.com/modelcontextprotocol/servers",
            "repository": "https://github.com/modelcontextprotocol/servers",
            "license": "MIT",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-jira"],
            "env": {
                "JIRA_URL": "https://your-domain.atlassian.net",
                "JIRA_EMAIL": "your-email@example.com",
                "JIRA_API_TOKEN": "your-api-token"
            },
            "capabilities": ["issue_management", "project_tracking", "task_creation"],
            "tags": ["jira", "project_management", "tasks", "productivity"],
            "discovery_source": DiscoverySource.MANUAL,
            "auto_start": False,
            "tools": [
                {
                    "name": "create_issue",
                    "description": "Create a new Jira issue",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "project": {"type": "string", "description": "Project key"},
                        "summary": {"type": "string", "description": "Issue summary"},
                        "description": {"type": "string", "description": "Issue description"},
                        "issue_type": {"type": "string", "description": "Issue type (Bug, Task, Story)"}
                    }
                },
                {
                    "name": "search_issues",
                    "description": "Search for Jira issues using JQL",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "jql": {"type": "string", "description": "JQL query string"}
                    }
                }
            ]
        },
        {
            "name": "OpenAI MCP Server",
            "description": "OpenAI API integration - GPT models, embeddings, completions",
            "version": "1.0.0",
            "author": "Community",
            "homepage": "https://github.com/modelcontextprotocol/servers",
            "repository": "https://github.com/modelcontextprotocol/servers",
            "license": "MIT",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-openai"],
            "env": {
                "OPENAI_API_KEY": "your-api-key"
            },
            "capabilities": ["text_generation", "embeddings", "chat_completion"],
            "tags": ["openai", "ai", "gpt", "llm"],
            "discovery_source": DiscoverySource.MANUAL,
            "auto_start": False,
            "tools": [
                {
                    "name": "chat_completion",
                    "description": "Generate a chat completion using GPT",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "messages": {"type": "array", "description": "Chat messages"},
                        "model": {"type": "string", "description": "Model name (e.g., gpt-4)"}
                    }
                },
                {
                    "name": "create_embedding",
                    "description": "Create text embeddings",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "text": {"type": "string", "description": "Text to embed"}
                    }
                }
            ]
        },
        {
            "name": "Redis MCP Server",
            "description": "Redis cache operations - get, set, delete keys, manage data structures",
            "version": "1.0.0",
            "author": "Community",
            "homepage": "https://github.com/modelcontextprotocol/servers",
            "repository": "https://github.com/modelcontextprotocol/servers",
            "license": "MIT",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-redis"],
            "env": {
                "REDIS_URL": "redis://localhost:6379"
            },
            "capabilities": ["caching", "key_value_storage", "data_structures"],
            "tags": ["redis", "cache", "database", "storage"],
            "discovery_source": DiscoverySource.MANUAL,
            "auto_start": False,
            "tools": [
                {
                    "name": "get",
                    "description": "Get value for a key",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "key": {"type": "string", "description": "Key name"}
                    }
                },
                {
                    "name": "set",
                    "description": "Set value for a key",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "key": {"type": "string", "description": "Key name"},
                        "value": {"type": "string", "description": "Value to store"}
                    }
                },
                {
                    "name": "delete",
                    "description": "Delete a key",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "key": {"type": "string", "description": "Key name"}
                    }
                }
            ]
        },
        {
            "name": "MongoDB MCP Server",
            "description": "MongoDB database operations - query, insert, update documents",
            "version": "1.0.0",
            "author": "Community",
            "homepage": "https://github.com/modelcontextprotocol/servers",
            "repository": "https://github.com/modelcontextprotocol/servers",
            "license": "MIT",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-mongodb"],
            "env": {
                "MONGODB_URI": "mongodb://localhost:27017/database"
            },
            "capabilities": ["document_storage", "query_execution", "aggregation"],
            "tags": ["mongodb", "database", "nosql", "documents"],
            "discovery_source": DiscoverySource.MANUAL,
            "auto_start": False,
            "tools": [
                {
                    "name": "find",
                    "description": "Find documents in a collection",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "collection": {"type": "string", "description": "Collection name"},
                        "query": {"type": "object", "description": "Query filter"}
                    }
                },
                {
                    "name": "insert",
                    "description": "Insert a document into a collection",
                    "tool_type": MCPToolType.FUNCTION,
                    "parameters": {
                        "collection": {"type": "string", "description": "Collection name"},
                        "document": {"type": "object", "description": "Document to insert"}
                    }
                }
            ]
        }
    ]

    # Create servers
    created_count = 0
    for server_data in default_servers:
        try:
            # Extract tools
            tools_data = server_data.pop("tools", [])
            
            # Create server
            server_create = MCPServerCreate(**server_data)
            server = await service.create_server(server_create)
            print(f"✅ Created server: {server.name} (ID: {server.server_id})")
            
            # Create tools for this server
            for tool_data in tools_data:
                tool_create = MCPToolBase(**tool_data)
                tool = await service.create_tool(tool_create, server.server_id)
                print(f"   └─ Added tool: {tool.name}")
            
            created_count += 1
            
        except Exception as e:
            print(f"❌ Error creating server {server_data.get('name')}: {e}")
    
    print("\n" + "=" * 60)
    print(f"✅ Successfully created {created_count} MCP servers")
    print("=" * 60)
    
    # List all servers
    all_servers = await service.list_servers(limit=1000)
    print(f"\n📊 Total MCP servers in database: {len(all_servers)}")
    for server in all_servers:
        print(f"   • {server.name} ({server.tool_count} tools)")

if __name__ == "__main__":
    asyncio.run(seed_mcp_servers())

