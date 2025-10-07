# GRETA - User Guide

## 📖 Quick Start Guide

### First Time Setup

1. **Start the Backend**
   ```bash
   cd backend
   python main.py
   ```

2. **Start the Frontend**
   ```bash
   cd greta-frontend
   npm start
   ```

3. **Access GRETA**
   - Open browser: `http://localhost:3000`
   - You should see the GRETA dashboard

---

## 🎯 Using Each Module

### 1. Master Agent Dashboard

**Location:** Main Dashboard

**What You Can Do:**
- View system health (CPU, memory, disk)
- Monitor all active agents
- Control system operations
- View activity logs
- Emergency controls

**Quick Actions:**
- **Pause All Agents:** Click "Pause System"
- **Resume Operations:** Click "Resume System"
- **View Metrics:** Check the metrics panel
- **Emergency Shutdown:** Use emergency controls

---

### 2. Building Your First Agent

**Location:** Agent Builder → Create New Project

**Step-by-Step:**

1. **Click "Create New Project"**
   - Enter project name
   - Choose template (Basic, Advanced, ML, LangChain)
   - Add description

2. **Configure Agent**
   - Edit `agent.py` file
   - Set configuration in `config.json`
   - Add dependencies if needed

3. **Build Agent**
   - Click "Build Project"
   - Wait for build to complete
   - Check build logs

4. **Test Agent**
   - Click "Run Tests"
   - Review test results
   - Check coverage metrics

5. **Deploy Agent**
   - Click "Deploy"
   - Select environment (development/production)
   - Confirm deployment

**Example Agent Code:**
```python
class MyAgent:
    def __init__(self):
        self.name = "Research Assistant"
        self.tools = ["web_search", "document_analysis"]
    
    async def execute(self, task):
        # Your agent logic here
        results = await self.search_web(task)
        analysis = await self.analyze(results)
        return analysis
```

---

### 3. Creating Workflows

**Location:** Autonomous Operations → Create Workflow

**Workflow Types:**

**Sequential Workflow:**
- Tasks execute one after another
- Use for: Data pipelines, step-by-step processes

**Parallel Workflow:**
- Tasks execute simultaneously
- Use for: Independent tasks, batch processing

**Conditional Workflow:**
- Tasks based on conditions
- Use for: Decision trees, branching logic

**Loop Workflow:**
- Repeat tasks until condition met
- Use for: Iterative processes, monitoring

**Scheduled Workflow:**
- Time-based execution
- Use for: Regular tasks, maintenance

**Creating a Workflow:**

1. Click "Create Workflow"
2. Choose workflow type
3. Add steps:
   ```json
   {
     "name": "Daily Report",
     "type": "sequential",
     "steps": [
       {"action": "collect_data", "params": {"source": "analytics"}},
       {"action": "generate_report", "params": {"format": "pdf"}},
       {"action": "send_email", "params": {"to": "team@company.com"}}
     ],
     "schedule": "0 9 * * *"  // Daily at 9 AM
   }
   ```
4. Click "Create"
5. Monitor execution in workflow dashboard

---

### 4. Using Predictive Analytics

**Location:** Predictive Analytics

**Creating a Prediction Model:**

1. **Prepare Your Data**
   - Click "Upload Dataset"
   - Select CSV or JSON file
   - Map columns (features and target)

2. **Choose Model Type**
   - **Time Series:** For forecasting (sales, stock prices)
   - **Classification:** For categories (spam/not spam)
   - **Regression:** For continuous values (price prediction)
   - **Clustering:** For grouping (customer segments)
   - **Anomaly Detection:** For outliers (fraud detection)

3. **Train Model**
   - Set hyperparameters
   - Click "Train Model"
   - Wait for training to complete

4. **Make Predictions**
   - Input new data
   - Click "Predict"
   - View results

**Example Use Cases:**
- Sales forecasting
- Customer churn prediction
- Demand planning
- Anomaly detection
- Sentiment analysis

---

### 5. Managing Agents

**Location:** Agent Management

**Key Features:**

**Agent Registry:**
- View all agents
- Filter by type, status
- Search by name

**Version Control:**
- Create new versions
- Compare versions
- Rollback to previous version

**Performance Monitoring:**
- Response times
- Success rates
- Error rates
- User satisfaction

**A/B Testing:**
- Create test (Agent A vs Agent B)
- Define metrics
- Run test
- Analyze winner

**Example A/B Test:**
```python
{
  "name": "Response Quality Test",
  "agent_a_id": "agent_v1",
  "agent_b_id": "agent_v2",
  "metrics": ["response_time", "accuracy"],
  "duration_days": 7
}
```

---

### 6. Using MCP Tools

**Location:** MCP Hub

**Available Tools:**

1. **Web Search** - Search the internet
2. **ArXiv** - Research papers
3. **Image Search** - Find images
4. **Screenshot** - Capture screenshots
5. **File Operations** - Read/write files
6. **Database Query** - Query databases
7. **API Calls** - Call external APIs
8. **Code Execution** - Run code safely
9. **Data Analysis** - Analyze data
10. **Document Processing** - Process documents
11. **Email** - Send emails
12. **Calendar** - Manage calendar
13. **Task Automation** - Automate tasks
14. **System Commands** - Run system commands

**Using a Tool:**

1. Select tool from list
2. Configure parameters
3. Click "Execute"
4. View results

**Example - Web Search:**
```python
{
  "tool": "web_search",
  "params": {
    "query": "latest AI research",
    "num_results": 10
  }
}
```

---

### 7. Memory Management

**Location:** Functional Memory

**Memory Types:**

**Episodic Memory:**
- Stores specific events
- Use for: Conversation history, user interactions

**Semantic Memory:**
- Stores facts and knowledge
- Use for: General knowledge, definitions

**Procedural Memory:**
- Stores how-to knowledge
- Use for: Processes, procedures

**Working Memory:**
- Temporary storage
- Use for: Active tasks, current context

**Operations:**

**Store Memory:**
```python
{
  "content": "User prefers detailed explanations",
  "type": "semantic",
  "importance": 0.8,
  "tags": ["user_preference", "communication"]
}
```

**Search Memory:**
```python
{
  "query": "user preferences",
  "type": "semantic",
  "limit": 10
}
```

**Consolidate Memory:**
- Merges similar memories
- Removes duplicates
- Strengthens important memories

---

### 8. OSINT Investigations

**Location:** OSINT Platform

**Creating an Investigation:**

1. Click "New Investigation"
2. Enter details:
   - Title
   - Description
   - Priority (low/medium/high)
3. Select data sources:
   - Web
   - Databases
   - Technical
   - Social Media
   - Geospatial
4. Click "Start Investigation"

**Gathering Data:**
- Click "Gather Data"
- Select sources
- Configure parameters
- View collected data

**Analyzing Data:**
- Click "Analyze"
- Review confidence scores
- Check insights
- Export report

**Example Investigation:**
```python
{
  "title": "Competitor Analysis",
  "description": "Analyze competitor products and pricing",
  "sources": ["web", "social"],
  "priority": "high",
  "keywords": ["competitor", "pricing", "features"]
}
```

---

### 9. Multimodal Processing

**Location:** Multimodal Processing

**Processing Types:**

**Image Processing:**
- OCR (text extraction)
- Object detection
- Image analysis
- Face recognition

**Audio Processing:**
- Transcription
- Speaker identification
- Sentiment analysis
- Audio enhancement

**Video Processing:**
- Frame extraction
- Scene detection
- Object tracking
- Video summarization

**Document Processing:**
- Text extraction
- Summarization
- Entity recognition
- Classification

**Example - Process Image:**
1. Upload image or provide URL
2. Select operations (OCR, object detection)
3. Click "Process"
4. View results

---

### 10. Knowledge Base

**Location:** Knowledge Base

**Managing Documents:**

**Add Document:**
1. Click "Add Document"
2. Enter title and content
3. Add tags
4. Select category
5. Click "Save"

**Search Documents:**
- Use semantic search
- Filter by category
- Filter by tags
- Sort by relevance

**Knowledge Graph:**
- View document relationships
- Explore connections
- Find related documents

**Example Document:**
```python
{
  "title": "API Integration Guide",
  "content": "How to integrate with external APIs...",
  "category": "technical",
  "tags": ["api", "integration", "guide"],
  "metadata": {
    "author": "John Doe",
    "version": "1.0"
  }
}
```

---

### 11. Training Platform

**Location:** Training Platform

**Creating a Course:**

1. Click "Create Course"
2. Enter course details:
   - Title
   - Description
   - Category
   - Difficulty level
3. Add modules/lessons
4. Create assessments
5. Publish course

**Enrolling Students:**
1. Select course
2. Click "Enroll User"
3. Enter user ID
4. Confirm enrollment

**Tracking Progress:**
- View enrollment details
- Check completion percentage
- Review assessment scores
- Issue certificates

**Creating Assessments:**
```python
{
  "title": "Module 1 Quiz",
  "questions": [
    {
      "question": "What is an AI agent?",
      "options": ["A", "B", "C", "D"],
      "correct": "A"
    }
  ],
  "passing_score": 70,
  "duration_minutes": 30
}
```

---

## 🔧 Common Tasks

### Task 1: Daily System Check

1. Open Master Agent Dashboard
2. Check system health metrics
3. Review active agents
4. Check for errors in activity log
5. Verify all services running

### Task 2: Deploy New Agent

1. Build agent in Agent Builder
2. Run tests
3. Review test results
4. Deploy to staging
5. Test in staging
6. Deploy to production
7. Monitor performance

### Task 3: Create Automated Report

1. Create workflow in Autonomous Operations
2. Add data collection step
3. Add analysis step
4. Add report generation step
5. Add email delivery step
6. Schedule workflow
7. Monitor execution

### Task 4: Research Task

1. Create investigation in OSINT
2. Configure data sources
3. Gather data
4. Analyze results
5. Export report
6. Store in Knowledge Base

### Task 5: Train Prediction Model

1. Prepare dataset
2. Upload to Predictive Analytics
3. Select model type
4. Configure hyperparameters
5. Train model
6. Evaluate performance
7. Make predictions

---

## 💡 Tips and Tricks

### Performance Tips

1. **Use Caching:** Enable caching for frequently accessed data
2. **Batch Operations:** Group similar requests
3. **Async Processing:** Use async for long-running tasks
4. **Monitor Resources:** Keep an eye on CPU/memory usage
5. **Regular Cleanup:** Remove old data and logs

### Best Practices

1. **Version Everything:** Use version control for agents and workflows
2. **Test Thoroughly:** Always test before deploying
3. **Monitor Continuously:** Set up alerts for errors
4. **Document Well:** Add descriptions and comments
5. **Backup Regularly:** Backup database and configurations

### Troubleshooting

**Agent Not Responding:**
- Check agent status in Agent Management
- Review error logs
- Restart agent if needed

**Workflow Failed:**
- Check workflow logs
- Verify all steps configured correctly
- Test individual steps

**Slow Performance:**
- Check system resources
- Reduce concurrent operations
- Optimize queries

**Data Not Saving:**
- Check database connection
- Verify permissions
- Check disk space

---

## 📞 Getting Help

### Resources

- **Documentation:** Full API docs at `/api/docs`
- **Training Manual:** Detailed guides and tutorials
- **GitHub:** Issues and discussions
- **Community:** Forums and chat

### Support Channels

- **GitHub Issues:** Bug reports and feature requests
- **Email:** support@greta-ai.com
- **Documentation:** https://github.com/Sounddeck/Greta/wiki

---

**Happy building with GRETA! 🚀**

