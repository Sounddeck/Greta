# Greta/CPAS4 Enhancement Analysis & Recommendations

**Date:** October 12, 2025  
**Project:** Greta/CPAS4 Personal AI Assistant  
**Phase:** Post-Phase 5 (Production Ready)  
**Analysis Type:** Comprehensive Feature Enhancement Assessment  

---

## Executive Summary

This document provides a detailed analysis of 12 requested enhancements for the Greta/CPAS4 system. Each enhancement has been evaluated for:
- Current implementation status
- Required modifications
- Implementation complexity
- Priority recommendations
- Dependencies and prerequisites

**Overall Assessment:**
- ✅ **7 features** have partial/foundational implementation
- ⚠️ **3 features** require moderate development
- 🔴 **2 features** require significant new development

---

## Enhancement Analysis

### 1. Prompt Library Integration ⚠️ MODERATE EFFORT

**Current Status:** Partial Implementation

**What Exists:**
- ✅ Prompt enhancement endpoint at `/api/v1/prompts/enhance`
- ✅ Agent-specific enhancement strategies (research, coding, data_analysis, etc.)
- ✅ Quality metrics calculation
- ✅ LLM-powered prompt improvement
- ✅ Prompt orchestrator service with master/context prompts
- ✅ Evolution prompt manager with versioning

**What's Missing:**
- ❌ User-facing prompt template library
- ❌ Template categorization system
- ❌ User-customizable templates
- ❌ Template CRUD endpoints
- ❌ Frontend UI for template management

**Files to Modify:**
```
Backend:
- backend/routers/prompts_router.py (add template endpoints)
- backend/services/prompt_library_service.py (NEW - create this)
- backend/models/prompt_models.py (NEW - add template models)

Frontend:
- cpas4-ui/src/components/prompts/ (add PromptLibrary.tsx)
- cpas4-ui/src/api/promptApi.ts (add template API calls)
- cpas4-ui/src/stores/promptStore.ts (NEW - Zustand store)
```

**Implementation Plan:**

**Phase 1: Backend Template System (2-3 hours)**
1. Create `PromptTemplate` model with fields:
   - `template_id`, `name`, `category`, `description`
   - `template_content`, `variables`, `tags`
   - `is_public`, `user_id`, `version`, `usage_count`
   - `created_at`, `updated_at`

2. Add endpoints to `prompts_router.py`:
   - `GET /api/v1/prompts/templates` - List templates
   - `GET /api/v1/prompts/templates/{id}` - Get template
   - `POST /api/v1/prompts/templates` - Create template
   - `PUT /api/v1/prompts/templates/{id}` - Update template
   - `DELETE /api/v1/prompts/templates/{id}` - Delete template
   - `POST /api/v1/prompts/templates/{id}/use` - Use template

3. Seed database with default templates:
   - Research prompts (academic, market, competitive)
   - Coding prompts (debugging, refactoring, documentation)
   - Content creation (blog, email, social media)
   - Data analysis (visualization, insights, reporting)

**Phase 2: Frontend UI (2-3 hours)**
1. Create `PromptLibrary.tsx` component with:
   - Template browser with category filters
   - Search functionality
   - Template preview
   - Variable input form
   - "Use Template" button integration with chat

2. Add navigation link in LeftSidebar.tsx
3. Create Zustand store for template state management

**Estimated Effort:** 4-6 hours  
**Priority:** ⭐⭐⭐ HIGH (Enhances core prompt functionality)  
**Dependencies:** None  

---

### 2. System Resource Monitoring UI ⚠️ MODERATE EFFORT

**Current Status:** Not Implemented

**What Exists:**
- ✅ Performance monitoring utilities in backend
- ✅ Metrics collection in various services
- ❌ No real-time system resource monitoring
- ❌ No frontend UI for resource display

**Implementation Plan:**

**Backend (1-2 hours):**
1. Create `/api/v1/system/resources` endpoint:
```python
# backend/routers/system_router.py
@router.get("/resources")
async def get_system_resources():
    import psutil
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "gpu_percent": get_gpu_usage(),  # If GPU available
        "disk_percent": psutil.disk_usage('/').percent,
        "timestamp": datetime.utcnow().isoformat()
    }
```

2. Add WebSocket endpoint for real-time updates (optional)

**Frontend (2-3 hours):**
1. Create `SystemStatusBar.tsx` component:
```typescript
// Slim status bar at bottom of screen
// Shows: CPU: 45% | Memory: 62% | GPU: 30%
// Color coding: Green (<70%), Yellow (70-85%), Red (>85%)
// Updates every 2 seconds
```

2. Add to `App.tsx` layout:
```typescript
<div className="h-screen flex flex-col">
  <Header />
  <div className="flex-1 flex overflow-hidden">
    {/* Main content */}
  </div>
  <SystemStatusBar /> {/* NEW */}
</div>
```

**Files to Create/Modify:**
```
Backend:
- backend/routers/system_router.py (NEW)
- backend/main.py (register router)

Frontend:
- cpas4-ui/src/components/system/SystemStatusBar.tsx (NEW)
- cpas4-ui/src/App.tsx (add status bar)
- cpas4-ui/src/api/systemApi.ts (NEW)
```

**Estimated Effort:** 3-5 hours  
**Priority:** ⭐⭐ MEDIUM (Nice-to-have, not critical)  
**Dependencies:** `psutil` Python package (likely already installed)  

---

### 3. File Attachment Debugging 🔴 INVESTIGATION REQUIRED

**Current Status:** Backend Working, Frontend Integration Issue

**What Exists:**
- ✅ Backend file upload endpoint: `/api/v1/knowledge/upload`
- ✅ File processing and storage logic
- ✅ Multiple upload endpoints (knowledge, web_commands, asset_creation)
- ⚠️ Frontend attachment UI may not be connected

**Investigation Steps:**

1. **Check Frontend Attachment Component:**
```bash
# Search for file upload UI in frontend
find cpas4-ui/src -name "*.tsx" -exec grep -l "file.*upload\|attachment" {} \;
```

2. **Verify API Integration:**
```typescript
// Check if MasterAgentContext.tsx has file upload
// Look for: <input type="file"> or FileUpload component
```

3. **Test Backend Endpoint:**
```bash
curl -X POST http://localhost:8000/api/v1/knowledge/upload \
  -F "files=@test.txt" \
  -F "collection=test"
```

**Likely Issues:**
- Frontend file input not connected to API
- Missing FormData handling in API client
- CORS issues with file uploads
- Missing file upload UI component

**Fix Plan (2-4 hours):**
1. Add file upload button to chat input area
2. Implement FormData API call
3. Show uploaded files in context window
4. Add file preview/remove functionality

**Files to Check/Modify:**
```
Frontend:
- cpas4-ui/src/components/agents/MasterAgentContext.tsx
- cpas4-ui/src/lib/api.ts (add file upload method)
- cpas4-ui/src/components/shared/FileUpload.tsx (NEW)
```

**Estimated Effort:** 2-4 hours (after investigation)  
**Priority:** ⭐⭐⭐ HIGH (Core functionality)  
**Dependencies:** None  

---

### 4. Database Backup Strategy ✅ DOCUMENTATION NEEDED

**Current Status:** MongoDB Standard Backup Available

**What Exists:**
- ✅ MongoDB database: `greta_pai`
- ✅ Standard MongoDB backup tools available
- ❌ No automated backup scripts
- ❌ No backup documentation

**Manual Backup Commands:**

**1. Full Database Backup:**
```bash
# Backup entire greta_pai database
mongodump --db=greta_pai --out=/path/to/backup/$(date +%Y%m%d_%H%M%S)

# Example:
mongodump --db=greta_pai --out=~/backups/greta_$(date +%Y%m%d_%H%M%S)
```

**2. Restore from Backup:**
```bash
# Restore database
mongorestore --db=greta_pai /path/to/backup/greta_pai

# Example:
mongorestore --db=greta_pai ~/backups/greta_20251012_140000/greta_pai
```

**3. Specific Collection Backup:**
```bash
# Backup single collection
mongodump --db=greta_pai --collection=interactions --out=~/backups/

# Restore single collection
mongorestore --db=greta_pai --collection=interactions ~/backups/greta_pai/interactions.bson
```

**Automated Backup Script:**

Create `backend/scripts/backup_database.sh`:
```bash
#!/bin/bash
# Greta Database Backup Script

BACKUP_DIR=~/greta_backups
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH=$BACKUP_DIR/greta_$DATE

# Create backup directory
mkdir -p $BACKUP_DIR

# Perform backup
mongodump --db=greta_pai --out=$BACKUP_PATH

# Compress backup
tar -czf $BACKUP_PATH.tar.gz -C $BACKUP_DIR greta_$DATE

# Remove uncompressed backup
rm -rf $BACKUP_PATH

# Keep only last 7 backups
ls -t $BACKUP_DIR/*.tar.gz | tail -n +8 | xargs rm -f

echo "✅ Backup completed: $BACKUP_PATH.tar.gz"
```

**Automated Scheduling (macOS):**

Create `~/Library/LaunchAgents/com.greta.backup.plist`:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.greta.backup</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/macone/CPAS/Desktop/CPAS5/backend/scripts/backup_database.sh</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>2</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
</dict>
</plist>
```

Load with: `launchctl load ~/Library/LaunchAgents/com.greta.backup.plist`

**Estimated Effort:** 1-2 hours (documentation + script creation)  
**Priority:** ⭐⭐⭐⭐ CRITICAL (Data protection)  
**Dependencies:** MongoDB installed  

---

### 5. MCP Server & Tool Selection ✅ PARTIALLY IMPLEMENTED

**Current Status:** Backend Complete, Frontend Needs Enhancement

**What Exists:**
- ✅ MCP server management: `/api/v1/mcp/servers`
- ✅ MCP tool management: `/api/v1/mcp/tools`
- ✅ Server CRUD operations
- ✅ Tool discovery and execution
- ✅ Frontend MCPManager component
- ⚠️ No per-agent MCP configuration

**What's Missing:**
- ❌ Agent-specific MCP server/tool selection
- ❌ UI in Agent Builder for MCP configuration
- ❌ Database schema for agent-MCP relationships

**Implementation Plan (3-4 hours):**

**Backend:**
1. Add `mcp_config` field to Agent model:
```python
class Agent(BaseModel):
    # ... existing fields ...
    mcp_config: Optional[Dict[str, Any]] = {
        "enabled_servers": [],  # List of server IDs
        "enabled_tools": [],    # List of tool IDs
        "tool_permissions": {}  # Tool-specific permissions
    }
```

2. Add endpoint: `PUT /api/v1/agents/{id}/mcp-config`

**Frontend:**
1. Add MCP Configuration step to Agent Builder:
```typescript
// Step 4: MCP Configuration
<MCPConfigurationStep
  availableServers={mcpServers}
  availableTools={mcpTools}
  selectedServers={agent.mcp_config.enabled_servers}
  selectedTools={agent.mcp_config.enabled_tools}
  onUpdate={handleMCPConfigUpdate}
/>
```

**Files to Modify:**
```
Backend:
- backend/models/agent_builder_models.py (add mcp_config)
- backend/routers/agent_builder_router.py (add MCP config endpoint)

Frontend:
- cpas4-ui/src/components/agent-builder/MCPConfigurationStep.tsx (NEW)
- cpas4-ui/src/components/agent-builder/AgentBuilderWorkflow.tsx (add step)
```

**Estimated Effort:** 3-4 hours  
**Priority:** ⭐⭐⭐ HIGH (Enhances agent customization)  
**Dependencies:** None  

---

## Summary Table

| # | Enhancement | Status | Effort | Priority | Dependencies |
|---|-------------|--------|--------|----------|--------------|
| 1 | Prompt Library | ⚠️ Partial | 4-6h | ⭐⭐⭐ | None |
| 2 | Resource Monitoring | 🔴 New | 3-5h | ⭐⭐ | psutil |
| 3 | File Attachment Fix | ⚠️ Debug | 2-4h | ⭐⭐⭐ | None |
| 4 | Database Backup | ✅ Docs | 1-2h | ⭐⭐⭐⭐ | MongoDB |
| 5 | MCP Per-Agent Config | ⚠️ Partial | 3-4h | ⭐⭐⭐ | None |
| 6 | Google Multi-Account | 🔴 New | 6-8h | ⭐⭐⭐ | OAuth setup |
| 7 | Lead Generation | 🔴 New | 8-12h | ⭐⭐ | OSINT tools |
| 8 | Continuous Learning | ✅ Exists | 0h | N/A | Active |
| 9 | Data Visualization | ✅ Exists | 0h | N/A | Active |
| 10 | Advanced Nav Links | ✅ Exists | 0h | N/A | Complete |
| 11 | Code Generation | ✅ Exists | 0h | N/A | Active |
| 12 | OSINT Toolkit | ✅ Exists | 0h | N/A | Active |

**Total Estimated Effort for New Features:** 27-41 hours

---

## Next Steps

**Immediate Actions (High Priority):**
1. ✅ Create database backup script and documentation
2. 🔧 Debug and fix file attachment functionality
3. 📚 Implement prompt template library

**Short-term (Medium Priority):**
4. 🎛️ Add per-agent MCP configuration
5. 📊 Implement system resource monitoring

**Long-term (Lower Priority):**
6. 🌐 Add Google Workspace multi-account support
7. 🔍 Build lead generation research module

---

**Document continues in ENHANCEMENT_ANALYSIS_PART2.md...**

