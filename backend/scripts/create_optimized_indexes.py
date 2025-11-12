"""
Create Optimized Database Indexes

Phase 3, Task 3.1: Optimize Database Queries
This script creates comprehensive indexes for all collections based on query patterns.

Run this script to create/update all database indexes:
    python -m backend.scripts.create_optimized_indexes
"""

import asyncio
import logging
from database import Database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_all_indexes():
    """Create all optimized indexes for the database"""
    db = Database()
    await db.connect()
    
    if not db.connected or db.db is None:
        logger.error("❌ Database not connected")
        return
    
    logger.info("🔧 Creating optimized indexes...")
    
    # Agent Builder indexes
    await create_agent_builder_indexes(db)
    
    # CRM indexes
    await create_crm_indexes(db)
    
    # ATS indexes
    await create_ats_indexes(db)
    
    # MCP indexes
    await create_mcp_indexes(db)
    
    # System indexes
    await create_system_indexes(db)
    
    logger.info("✅ All indexes created successfully!")
    await db.disconnect()


async def create_agent_builder_indexes(db: Database):
    """Create indexes for Agent Builder collections"""
    logger.info("📊 Creating Agent Builder indexes...")
    
    # agent_templates
    collection = db.db["agent_templates"]
    await collection.create_index("template_id", unique=True)
    await collection.create_index("category")
    await collection.create_index("created_at")
    await collection.create_index([("category", 1), ("created_at", -1)])
    
    # custom_agents
    collection = db.db["custom_agents"]
    await collection.create_index("agent_id", unique=True)
    await collection.create_index("status")
    await collection.create_index("agent_type")
    await collection.create_index("created_at")
    await collection.create_index("updated_at")
    await collection.create_index([("status", 1), ("created_at", -1)])
    await collection.create_index([("agent_type", 1), ("status", 1)])
    await collection.create_index("tags")  # Multikey index for array
    
    # agent_versions
    collection = db.db["agent_versions"]
    await collection.create_index("version_id", unique=True)
    await collection.create_index("agent_id")
    await collection.create_index("created_at")
    await collection.create_index([("agent_id", 1), ("created_at", -1)])
    
    # agent_deployments
    collection = db.db["agent_deployments"]
    await collection.create_index("deployment_id", unique=True)
    await collection.create_index("agent_id")
    await collection.create_index("status")
    await collection.create_index("started_at")
    await collection.create_index([("agent_id", 1), ("started_at", -1)])
    await collection.create_index([("status", 1), ("started_at", -1)])
    
    # agent_executions
    collection = db.db["agent_executions"]
    await collection.create_index("execution_id", unique=True)
    await collection.create_index("agent_id")
    await collection.create_index("status")
    await collection.create_index("started_at")
    await collection.create_index([("agent_id", 1), ("started_at", -1)])
    await collection.create_index([("status", 1), ("started_at", -1)])
    
    # agent_drafts
    collection = db.db["agent_drafts"]
    await collection.create_index("draft_id", unique=True)
    await collection.create_index("agent_id")
    await collection.create_index("created_at")
    await collection.create_index("updated_at")
    await collection.create_index([("agent_id", 1), ("updated_at", -1)])
    
    logger.info("✅ Agent Builder indexes created")


async def create_crm_indexes(db: Database):
    """Create indexes for CRM collections"""
    logger.info("📊 Creating CRM indexes...")
    
    # crm_contacts
    collection = db.db["crm_contacts"]
    await collection.create_index("contact_id", unique=True)
    await collection.create_index("email")
    await collection.create_index("company_id")
    await collection.create_index("status")
    await collection.create_index("created_at")
    await collection.create_index("updated_at")
    await collection.create_index([("status", 1), ("created_at", -1)])
    await collection.create_index([("company_id", 1), ("status", 1)])
    await collection.create_index("tags")  # Multikey index
    
    # crm_companies
    collection = db.db["crm_companies"]
    await collection.create_index("company_id", unique=True)
    await collection.create_index("name")
    await collection.create_index("industry")
    await collection.create_index("created_at")
    await collection.create_index([("industry", 1), ("created_at", -1)])
    await collection.create_index("tags")  # Multikey index
    
    # crm_deals
    collection = db.db["crm_deals"]
    await collection.create_index("deal_id", unique=True)
    await collection.create_index("contact_id")
    await collection.create_index("company_id")
    await collection.create_index("stage")
    await collection.create_index("created_at")
    await collection.create_index("expected_close_date")
    await collection.create_index([("stage", 1), ("created_at", -1)])
    await collection.create_index([("contact_id", 1), ("stage", 1)])
    await collection.create_index([("company_id", 1), ("stage", 1)])
    await collection.create_index([("expected_close_date", 1), ("stage", 1)])
    
    # crm_activities
    collection = db.db["crm_activities"]
    await collection.create_index("activity_id", unique=True)
    await collection.create_index("contact_id")
    await collection.create_index("deal_id")
    await collection.create_index("activity_type")
    await collection.create_index("created_at")
    await collection.create_index([("contact_id", 1), ("created_at", -1)])
    await collection.create_index([("deal_id", 1), ("created_at", -1)])
    await collection.create_index([("activity_type", 1), ("created_at", -1)])
    
    logger.info("✅ CRM indexes created")


async def create_ats_indexes(db: Database):
    """Create indexes for ATS collections"""
    logger.info("📊 Creating ATS indexes...")
    
    # ats_jobs
    collection = db.db["ats_jobs"]
    await collection.create_index("job_id", unique=True)
    await collection.create_index("status")
    await collection.create_index("job_type")
    await collection.create_index("department")
    await collection.create_index("created_at")
    await collection.create_index([("status", 1), ("created_at", -1)])
    await collection.create_index([("job_type", 1), ("status", 1)])
    await collection.create_index([("department", 1), ("status", 1)])
    await collection.create_index("tags")  # Multikey index
    
    # ats_candidates
    collection = db.db["ats_candidates"]
    await collection.create_index("candidate_id", unique=True)
    await collection.create_index("email")
    await collection.create_index("status")
    await collection.create_index("created_at")
    await collection.create_index("last_activity")
    await collection.create_index([("status", 1), ("created_at", -1)])
    await collection.create_index([("status", 1), ("last_activity", -1)])
    
    # ats_applications
    collection = db.db["ats_applications"]
    await collection.create_index("application_id", unique=True)
    await collection.create_index("job_id")
    await collection.create_index("candidate_id")
    await collection.create_index("status")
    await collection.create_index("created_at")
    await collection.create_index([("job_id", 1), ("status", 1)])
    await collection.create_index([("candidate_id", 1), ("created_at", -1)])
    await collection.create_index([("status", 1), ("created_at", -1)])
    
    # ats_interviews
    collection = db.db["ats_interviews"]
    await collection.create_index("interview_id", unique=True)
    await collection.create_index("application_id")
    await collection.create_index("candidate_id")
    await collection.create_index("status")
    await collection.create_index("scheduled_at")
    await collection.create_index([("application_id", 1), ("scheduled_at", -1)])
    await collection.create_index([("status", 1), ("scheduled_at", 1)])
    
    logger.info("✅ ATS indexes created")


async def create_mcp_indexes(db: Database):
    """Create indexes for MCP collections"""
    logger.info("📊 Creating MCP indexes...")
    
    # mcp_servers
    collection = db.db["mcp_servers"]
    await collection.create_index("server_id", unique=True)
    await collection.create_index("status")
    await collection.create_index("created_at")
    await collection.create_index([("status", 1), ("created_at", -1)])
    
    logger.info("✅ MCP indexes created")


async def create_system_indexes(db: Database):
    """Create indexes for system collections"""
    logger.info("📊 Creating system indexes...")
    
    # memory
    collection = db.db["memory"]
    await collection.create_index("timestamp")
    await collection.create_index("agent_id")
    await collection.create_index([("agent_id", 1), ("timestamp", -1)])
    
    # interactions
    collection = db.db["interactions"]
    await collection.create_index("timestamp")
    await collection.create_index("user_id")
    await collection.create_index("session_id")
    await collection.create_index([("user_id", 1), ("timestamp", -1)])
    
    # agents
    collection = db.db["agents"]
    await collection.create_index("agent_id")
    await collection.create_index("status")
    await collection.create_index("last_active")
    await collection.create_index([("status", 1), ("last_active", -1)])
    
    # performance
    collection = db.db["performance"]
    await collection.create_index("timestamp")
    await collection.create_index("metric_type")
    await collection.create_index("agent_id")
    await collection.create_index([("agent_id", 1), ("timestamp", -1)])
    
    logger.info("✅ System indexes created")


if __name__ == "__main__":
    asyncio.run(create_all_indexes())

