"""
CPAS4 UI Tasks Router
Provides task execution endpoints for the CPAS4 UI
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from database import get_database
from services.llm_service import get_llm_service
from services.enhanced_master_agent import EnhancedMasterAgent
from services.system_knowledge_service import SystemKnowledgeService
from services.data_extraction_service import DataExtractionService
from services.mongodb_query_service import MongoDBQueryService

logger = logging.getLogger(__name__)

# Initialize data services
data_extraction_service = DataExtractionService()
mongodb_query_service = MongoDBQueryService()

router = APIRouter(prefix="/api/v1/tasks", tags=["Tasks"])

# Global master agent instance
master_agent_instance: Optional[EnhancedMasterAgent] = None

# Pydantic models
class TaskExecuteRequest(BaseModel):
    """Task execution request from UI"""
    agent_id: str = Field(default="master_agent", description="Agent ID")
    description: str = Field(..., description="Task description/prompt")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Additional input data")
    max_iterations: int = Field(default=3, description="Maximum iterations")

class TaskResponse(BaseModel):
    """Task execution response"""
    id: str
    status: str
    result: Optional[str] = None
    output: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None

async def get_master_agent() -> EnhancedMasterAgent:
    """Get or initialize master agent instance"""
    global master_agent_instance

    if master_agent_instance is None:
        try:
            logger.info("🔧 Initializing master agent for the first time...")
            from services.learning_service import LearningService

            db = await get_database()
            llm_service = await get_llm_service()
            learning_service = LearningService(db.client)
            await learning_service.initialize()

            # Initialize system knowledge base
            logger.info("📚 Initializing system knowledge base...")
            knowledge_service = SystemKnowledgeService(db.db)
            await knowledge_service.initialize()
            logger.info("✅ System knowledge base initialized")

            master_agent_instance = EnhancedMasterAgent(db, llm_service, learning_service)
            await master_agent_instance.initialize()
            logger.info("✅ Master agent initialized successfully")
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"❌ Failed to initialize master agent: {e}")
            logger.error(f"   Full traceback:\n{error_trace}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize master agent: {str(e)}")
    else:
        logger.info("♻️  Reusing existing master agent instance")

    return master_agent_instance

@router.post("/execute")
async def execute_task(
    request: TaskExecuteRequest,
    master_agent: EnhancedMasterAgent = Depends(get_master_agent)
):
    """
    Execute a task using the master agent

    This endpoint is designed for the CPAS4 UI to execute tasks with different agent types.
    """
    try:
        # Extract agent type from input_data
        agent_type = request.input_data.get('agent_type', 'general')
        conversation_history = request.input_data.get('conversation_history', [])
        uploaded_files = request.input_data.get('uploaded_files', [])

        logger.info(f"📥 Task execution request received")
        logger.info(f"   Agent type: {agent_type}")
        logger.info(f"   Description: {request.description[:100]}...")
        logger.info(f"   Conversation history length: {len(conversation_history)}")
        logger.info(f"   Uploaded files: {len(uploaded_files)}")
        logger.info(f"   Agent ID: {request.agent_id}")
        logger.info(f"   Max iterations: {request.max_iterations}")
        
        # Use the LLM service to generate a response
        llm_service = await get_llm_service()
        
        # Build context from conversation history
        context = ""
        if conversation_history:
            context = "\n".join([
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in conversation_history[-5:]  # Last 5 messages
            ])

        # Build file context from uploaded files
        file_context = ""
        if uploaded_files:
            file_list = "\n".join([
                f"- {file.get('filename', 'unknown')} ({file.get('file_size', 0)} bytes)"
                for file in uploaded_files
            ])
            file_context = f"\n\nUploaded files available for analysis:\n{file_list}"
            logger.info(f"📎 Files available for processing:\n{file_context}")
        
        # Create agent-specific system prompt
        system_prompts = {
            'research': "You are a Research Agent specialized in conducting comprehensive research, gathering information, and analyzing data from multiple sources. Provide detailed, well-researched responses with citations when possible.",
            'coding': "You are a Coding Agent expert in software development, code generation, debugging, and technical problem-solving. Provide clean, well-documented code with explanations.",
            'data_analysis': "You are a Data Analysis Agent specialized in data processing, statistical analysis, visualization, and insights generation. Provide clear analytical insights with data-driven recommendations.",
            'content_creation': "You are a Content Creation Agent expert in creating engaging content, copywriting, and creative writing across various formats. Provide compelling, well-structured content.",
            'task_automation': "You are a Task Automation Agent specialized in automating workflows, creating scripts, and optimizing business processes. Provide practical automation solutions.",
            'customer_support': "You are a Customer Support Agent expert in customer service, issue resolution, and providing helpful support responses. Provide empathetic, solution-oriented responses.",
            'sales': "You are a Sales Agent specialized in sales strategies, lead generation, and customer relationship management. Provide persuasive, value-focused responses.",
            'marketing': "You are a Marketing Agent expert in marketing strategies, campaign planning, and brand development. Provide creative, strategic marketing insights.",
            'finance': "You are a Finance Agent specialized in financial analysis, budgeting, forecasting, and financial planning. Provide accurate, data-driven financial insights.",
            'general': "You are a versatile AI assistant capable of handling a wide range of tasks and queries. Provide helpful, accurate, and comprehensive responses."
        }
        
        system_prompt = system_prompts.get(agent_type, system_prompts['general'])

        # Get system knowledge context
        db = await get_database()
        knowledge_service = SystemKnowledgeService(db.db)
        system_knowledge = await knowledge_service.get_master_agent_context(max_entries=10)

        # Build the full prompt
        full_prompt = f"{system_prompt}\n\n"

        # Add system knowledge if available
        if system_knowledge:
            full_prompt += "## System Knowledge Base\n"
            full_prompt += system_knowledge + "\n\n"

        if context:
            full_prompt += f"Previous conversation:\n{context}\n\n"
        if file_context:
            full_prompt += file_context + "\n\n"
        full_prompt += f"User request: {request.description}"

        # Check if this is a file generation request
        description_lower = request.description.lower()
        file_gen_keywords = {
            'pdf': ['pdf', 'report', 'document'],
            'excel': ['excel', 'spreadsheet', 'xlsx', 'workbook'],
            'csv': ['csv', 'comma separated', 'data export'],
            'chart': ['chart', 'graph', 'visualization', 'plot', 'diagram']
        }

        file_gen_type = None
        for gen_type, keywords in file_gen_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                if 'generate' in description_lower or 'create' in description_lower or 'make' in description_lower:
                    file_gen_type = gen_type
                    break

        # If file generation detected, route to file generation service
        if file_gen_type:
            logger.info(f"📄 File generation request detected: {file_gen_type}")

            try:
                # Extract data requirements from the request
                data_requirements = data_extraction_service.extract_data_requirements(request.description)
                logger.info(f"📊 Data requirements: {data_requirements}")

                # Query MongoDB for real data if requirements specify a collection
                query_result = None
                use_real_data = data_requirements.get('has_real_data_requirement', False)

                if use_real_data:
                    logger.info("🔍 Querying MongoDB for real data...")
                    query_result = await mongodb_query_service.query_data(data_requirements)

                    if query_result.get('success') and query_result.get('count', 0) > 0:
                        logger.info(f"✅ Retrieved {query_result['count']} records from MongoDB")
                    else:
                        logger.warning(f"⚠️ No data found or query failed: {query_result.get('error', 'Unknown error')}")
                        logger.warning("⚠️ Falling back to sample data")
                        use_real_data = False

                # Generate file with real or sample data
                if file_gen_type == 'pdf':
                    if use_real_data and query_result and query_result.get('success'):
                        # Format data for PDF
                        formatted_data = mongodb_query_service.format_data_for_file(
                            query_result['data'],
                            query_result['headers'],
                            'pdf'
                        )
                        result_data = await master_agent.generate_pdf_report({
                            'title': request.description,
                            'content': formatted_data['content'],
                            'metadata': {
                                'author': 'CPAS5 Master Agent',
                                'generated_at': datetime.utcnow().isoformat(),
                                'data_source': query_result.get('collection', 'Unknown'),
                                'record_count': query_result.get('count', 0)
                            }
                        })
                    else:
                        # Use sample data
                        result_data = await master_agent.generate_pdf_report({
                            'title': request.description,
                            'content': [
                                {'type': 'paragraph', 'data': f'This is a sample PDF report generated based on your request: {request.description}'},
                                {'type': 'paragraph', 'data': 'The Master Agent can now generate PDF reports with custom content, tables, and formatting.'},
                                {'type': 'paragraph', 'data': '⚠️ Note: This report contains sample data. To generate reports with real data, specify a data source (e.g., "generate a PDF with contacts data").'}
                            ],
                            'metadata': {'author': 'CPAS5 Master Agent', 'generated_at': datetime.utcnow().isoformat()}
                        })

                elif file_gen_type == 'excel':
                    if use_real_data and query_result and query_result.get('success'):
                        # Format data for Excel
                        formatted_data = mongodb_query_service.format_data_for_file(
                            query_result['data'],
                            query_result['headers'],
                            'excel'
                        )
                        result_data = await master_agent.generate_excel_spreadsheet(formatted_data)
                    else:
                        # Use sample data
                        result_data = await master_agent.generate_excel_spreadsheet({
                            'sheets': [
                                {
                                    'name': 'Data',
                                    'headers': ['Item', 'Value', 'Status'],
                                    'data': [
                                        ['Sample 1', 100, 'Active'],
                                        ['Sample 2', 200, 'Pending'],
                                        ['Sample 3', 150, 'Complete'],
                                        ['⚠️ Note', 'Sample Data', 'Specify data source for real data']
                                    ]
                                }
                            ]
                        })

                elif file_gen_type == 'csv':
                    if use_real_data and query_result and query_result.get('success'):
                        # Format data for CSV
                        formatted_data = mongodb_query_service.format_data_for_file(
                            query_result['data'],
                            query_result['headers'],
                            'csv'
                        )
                        result_data = await master_agent.generate_csv_file(formatted_data)
                    else:
                        # Use sample data
                        result_data = await master_agent.generate_csv_file({
                            'headers': ['Name', 'Value', 'Category'],
                            'data': [
                                ['Item 1', '100', 'A'],
                                ['Item 2', '200', 'B'],
                                ['Item 3', '150', 'A'],
                                ['⚠️ Note', 'Sample Data', 'Specify data source']
                            ]
                        })

                elif file_gen_type == 'chart':
                    if use_real_data and query_result and query_result.get('success'):
                        # Format data for chart
                        formatted_data = mongodb_query_service.format_data_for_file(
                            query_result['data'],
                            query_result['headers'],
                            'chart'
                        )
                        formatted_data['title'] = request.description
                        result_data = await master_agent.generate_chart(formatted_data)
                    else:
                        # Use sample data
                        result_data = await master_agent.generate_chart({
                            'chart_type': 'bar',
                            'data': {
                                'x': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
                                'y': [10, 15, 13, 17, 20],
                                'x_label': 'Month',
                                'y_label': 'Value'
                            },
                            'title': 'Sample Chart (specify data source for real data)'
                        })

                if result_data.get('success'):
                    result = f"✅ {file_gen_type.upper()} file generated successfully!\n\n"

                    # Add data source information
                    if use_real_data and query_result and query_result.get('success'):
                        result += f"📊 Data Source: {query_result.get('collection', 'MongoDB')}\n"
                        result += f"📈 Records: {query_result.get('count', 0)}\n"
                        result += f"✅ Using REAL DATA from database\n\n"
                    else:
                        result += f"⚠️ Data Source: Sample/Placeholder Data\n"
                        result += f"💡 Tip: Specify a data source to use real data (e.g., 'with contacts data', 'from projects')\n\n"

                    result += f"📄 Filename: {result_data['filename']}\n"
                    result += f"📊 File size: {result_data['file_size']} bytes ({result_data['file_size']/1024:.2f} KB)\n"
                    result += f"📁 File type: {result_data['mime_type']}\n\n"
                    result += f"📂 Storage Location:\n"
                    result += f"   Full path: {result_data.get('absolute_path', result_data['filepath'])}\n"
                    result += f"   Directory: {result_data.get('storage_directory', 'generated_files/')}\n\n"
                    download_url = result_data.get('download_url', f'/api/v1/files/download/{result_data["filename"]}')
                    result += f"🔗 Download URL: {download_url}\n\n"
                    result += f"The file has been generated and is ready for download.\n"
                    result += f"You can download it using the URL above or find it at the full path shown."
                else:
                    result = f"❌ Failed to generate {file_gen_type} file: {result_data.get('error', 'Unknown error')}"

                task_id = f"task_{datetime.utcnow().timestamp()}"
                return {
                    "id": task_id,
                    "status": "completed",
                    "result": result,
                    "output": result,
                    "file_data": result_data if result_data.get('success') else None,
                    "created_at": datetime.utcnow().isoformat(),
                    "completed_at": datetime.utcnow().isoformat(),
                    "agent_type": agent_type,
                    "description": request.description
                }
            except Exception as file_gen_error:
                logger.error(f"❌ File generation failed: {file_gen_error}")
                result = f"Failed to generate {file_gen_type} file: {str(file_gen_error)}"

        # Log the full prompt being sent
        logger.info(f"📝 Full prompt being sent to LLM:")
        logger.info(f"   System prompt: {system_prompt[:100]}...")
        logger.info(f"   User request: {request.description}")
        logger.info(f"   Full prompt length: {len(full_prompt)} chars")

        # Query the LLM using generate_completion
        try:
            llm_request = {
                "message": full_prompt,
                "max_tokens": 2000,
                "temperature": 0.7,
                "agent_type": agent_type
            }

            logger.info(f"🤖 Calling LLM service...")
            response = await llm_service.generate_completion(llm_request)
            logger.info(f"✅ LLM response received")
            logger.info(f"   Response keys: {list(response.keys())}")
            logger.info(f"   Success: {response.get('success')}")
            logger.info(f"   Model used: {response.get('model_used')}")

            result = response.get('response', response.get('text', 'No response generated'))
            logger.info(f"📤 Final result length: {len(result)} chars")
            logger.info(f"   First 200 chars: {result[:200]}...")

        except Exception as llm_error:
            logger.error(f"LLM query failed: {llm_error}")
            # Fallback response with agent-specific context
            result = f"I'm the {agent_type.replace('_', ' ').title()} Agent. I received your request: '{request.description}'. However, I'm currently unable to process this request due to a technical issue. Please try again or contact support if the problem persists."
        
        # Create task response
        task_id = f"task_{datetime.utcnow().timestamp()}"
        
        return {
            "id": task_id,
            "status": "completed",
            "result": result,
            "output": result,
            "created_at": datetime.utcnow().isoformat(),
            "completed_at": datetime.utcnow().isoformat(),
            "agent_type": agent_type,
            "description": request.description
        }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"❌ Task execution failed: {e}")
        logger.error(f"   Error type: {type(e).__name__}")
        logger.error(f"   Full traceback:\n{error_trace}")
        raise HTTPException(
            status_code=500,
            detail=f"Task execution failed: {str(e)}"
        )

@router.get("/{task_id}")
async def get_task(task_id: str):
    """Get task status and result"""
    # This is a simplified implementation
    # In a real system, you would store tasks in a database
    return {
        "id": task_id,
        "status": "completed",
        "message": "Task retrieval not fully implemented yet"
    }

@router.get("/")
async def list_tasks():
    """List all tasks"""
    # This is a simplified implementation
    return {
        "tasks": [],
        "total": 0,
        "message": "Task listing not fully implemented yet"
    }

@router.post("/{task_id}/cancel")
async def cancel_task(task_id: str):
    """Cancel a running task"""
    # This is a simplified implementation
    return {
        "id": task_id,
        "status": "cancelled",
        "message": "Task cancellation not fully implemented yet"
    }

