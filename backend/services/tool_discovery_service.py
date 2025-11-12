"""
Tool Discovery Service for CPAS5 - Phase 2 Feature 1

This service provides progressive tool discovery capabilities, automatically
suggesting relevant tools based on task context, semantic matching, and
historical success patterns.
"""

import logging
import uuid
import re
import time
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from database import Database
from models.tool_discovery_models import (
    TaskAnalysis,
    TaskIntent,
    TaskComplexity,
    ToolSuggestion,
    ToolUsagePattern,
    ToolDiscoveryRequest,
    ToolDiscoveryResponse,
    ToolLearningData,
    ToolDiscoveryStats
)

logger = logging.getLogger(__name__)


class ToolDiscoveryService:
    """
    Progressive Tool Discovery Service
    
    This service analyzes tasks and suggests relevant tools based on:
    1. Keyword matching (30% weight)
    2. Semantic similarity (40% weight)
    3. Historical success rate (30% weight)
    
    The service learns from successful executions to improve suggestions over time.
    """
    
    def __init__(self, db: Database):
        self.db = db
        self.patterns_collection = "tool_usage_patterns"
        self.learning_collection = "tool_learning_data"
        self.stats_collection = "tool_discovery_stats"
        
        # Stop words for keyword extraction
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        # Intent keywords mapping
        self.intent_keywords = {
            TaskIntent.WEB_SCRAPING: ['scrape', 'extract', 'crawl', 'website', 'web', 'page', 'html'],
            TaskIntent.DATA_EXTRACTION: ['extract', 'parse', 'data', 'information', 'collect'],
            TaskIntent.RESEARCH: ['research', 'find', 'search', 'investigate', 'analyze', 'study'],
            TaskIntent.CODE_GENERATION: ['generate', 'create', 'write', 'code', 'function', 'class'],
            TaskIntent.CODE_ANALYSIS: ['analyze', 'review', 'debug', 'fix', 'optimize', 'code'],
            TaskIntent.DATA_ANALYSIS: ['analyze', 'statistics', 'data', 'metrics', 'trends'],
            TaskIntent.CONTENT_CREATION: ['write', 'create', 'generate', 'content', 'article', 'blog'],
            TaskIntent.AUTOMATION: ['automate', 'schedule', 'workflow', 'process', 'task'],
            TaskIntent.FILE_OPERATION: ['file', 'read', 'write', 'save', 'load', 'upload', 'download'],
            TaskIntent.API_INTERACTION: ['api', 'request', 'endpoint', 'call', 'fetch'],
            TaskIntent.DATABASE_OPERATION: ['database', 'query', 'sql', 'insert', 'update', 'delete']
        }
        
        logger.info("✅ Tool Discovery Service initialized")
    
    async def analyze_task(self, task_description: str) -> TaskAnalysis:
        """
        Analyze a task to extract keywords, entities, intent, and complexity
        
        Args:
            task_description: The task description to analyze
            
        Returns:
            TaskAnalysis object with extracted information
        """
        try:
            # Extract keywords
            keywords = self._extract_keywords(task_description)
            
            # Extract entities (URLs, file paths, etc.)
            entities = self._extract_entities(task_description)
            
            # Classify intent
            intent = self._classify_intent(task_description, keywords)
            
            # Estimate complexity
            complexity = self._estimate_complexity(task_description, keywords)
            
            # Determine if tools are likely needed
            requires_tools = self._requires_tools(task_description, intent)
            
            return TaskAnalysis(
                description=task_description,
                keywords=keywords,
                entities=entities,
                intent=intent,
                complexity=complexity,
                requires_tools=requires_tools
            )
            
        except Exception as e:
            logger.error(f"❌ Task analysis failed: {e}")
            # Return basic analysis on error
            return TaskAnalysis(
                description=task_description,
                keywords=[],
                entities=[],
                intent=TaskIntent.GENERAL,
                complexity=TaskComplexity.MEDIUM,
                requires_tools=True
            )
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Convert to lowercase and split
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        
        # Filter out stop words
        keywords = [w for w in words if w not in self.stop_words]
        
        # Count frequency
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top 15
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:15]]
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities like URLs, file paths, etc."""
        entities = []
        
        # Extract URLs
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, text)
        entities.extend(urls)
        
        # Extract file paths
        file_pattern = r'[a-zA-Z]:\\[^\s]+|/[^\s]+'
        files = re.findall(file_pattern, text)
        entities.extend(files)
        
        # Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        entities.extend(emails)
        
        return list(set(entities))[:10]  # Limit to 10 unique entities
    
    def _classify_intent(self, text: str, keywords: List[str]) -> TaskIntent:
        """Classify the task intent based on keywords"""
        text_lower = text.lower()
        keyword_set = set(keywords)
        
        # Score each intent
        intent_scores = {}
        for intent, intent_keywords in self.intent_keywords.items():
            score = 0
            for keyword in intent_keywords:
                if keyword in text_lower:
                    score += 2  # Exact match in text
                if keyword in keyword_set:
                    score += 1  # Match in extracted keywords
            intent_scores[intent] = score
        
        # Return intent with highest score
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            if best_intent[1] > 0:
                return best_intent[0]
        
        return TaskIntent.GENERAL
    
    def _estimate_complexity(self, text: str, keywords: List[str]) -> TaskComplexity:
        """Estimate task complexity based on various factors"""
        complexity_score = 0
        
        # Length of description
        if len(text) > 200:
            complexity_score += 2
        elif len(text) > 100:
            complexity_score += 1
        
        # Number of keywords
        if len(keywords) > 10:
            complexity_score += 2
        elif len(keywords) > 5:
            complexity_score += 1
        
        # Complexity indicators
        complex_words = ['multiple', 'complex', 'advanced', 'integrate', 'combine', 'analyze']
        for word in complex_words:
            if word in text.lower():
                complexity_score += 1
        
        # Classify based on score
        if complexity_score >= 4:
            return TaskComplexity.HIGH
        elif complexity_score >= 2:
            return TaskComplexity.MEDIUM
        else:
            return TaskComplexity.LOW
    
    def _requires_tools(self, text: str, intent: TaskIntent) -> bool:
        """Determine if task likely requires tools"""
        # Some intents always require tools
        tool_required_intents = {
            TaskIntent.WEB_SCRAPING,
            TaskIntent.DATA_EXTRACTION,
            TaskIntent.FILE_OPERATION,
            TaskIntent.API_INTERACTION,
            TaskIntent.DATABASE_OPERATION
        }
        
        if intent in tool_required_intents:
            return True
        
        # Check for tool-related keywords
        tool_keywords = ['scrape', 'extract', 'fetch', 'download', 'upload', 'api', 'database']
        text_lower = text.lower()
        for keyword in tool_keywords:
            if keyword in text_lower:
                return True
        
        return False
    
    async def discover_tools(
        self,
        request: ToolDiscoveryRequest
    ) -> ToolDiscoveryResponse:
        """
        Discover relevant tools for a task
        
        Args:
            request: Tool discovery request with task description and parameters
            
        Returns:
            ToolDiscoveryResponse with suggested tools and analysis
        """
        start_time = time.time()
        
        try:
            # Analyze the task
            task_analysis = await self.analyze_task(request.task_description)
            
            # Get all available tools from the registry
            from services.tool_registry import tool_registry

            # Ensure tool registry is initialized
            if not tool_registry.db:
                await tool_registry.initialize()

            # Get all tools from the registry's tools dictionary
            all_tools = list(tool_registry.tools.values())

            logger.info(f"🔍 Analyzing {len(all_tools)} tools for task: {request.task_description[:50]}...")
            
            # Score each tool
            scored_tools = []
            for tool in all_tools:
                score = await self._score_tool_relevance(
                    tool,
                    task_analysis,
                    request.agent_id,
                    request.context
                )
                
                if score.relevance_score >= request.min_relevance:
                    scored_tools.append(score)
            
            # Sort by relevance score
            scored_tools.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Apply ranking adjustments
            ranked_tools = await self._rank_tools(
                scored_tools[:request.limit * 2],  # Get more for ranking
                request.agent_id,
                request.context
            )
            
            # Return top N
            suggestions = ranked_tools[:request.limit]
            
            discovery_time = time.time() - start_time
            
            logger.info(f"✅ Discovered {len(suggestions)} tools in {discovery_time:.2f}s")
            
            # Update stats
            await self._update_discovery_stats(len(all_tools), len(suggestions), discovery_time)
            
            return ToolDiscoveryResponse(
                task_analysis=task_analysis,
                suggestions=suggestions,
                discovery_time=discovery_time,
                total_tools_analyzed=len(all_tools)
            )
            
        except Exception as e:
            logger.error(f"❌ Tool discovery failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Return empty response on error
            return ToolDiscoveryResponse(
                task_analysis=TaskAnalysis(
                    description=request.task_description,
                    keywords=[],
                    entities=[],
                    intent=TaskIntent.GENERAL,
                    complexity=TaskComplexity.MEDIUM,
                    requires_tools=True
                ),
                suggestions=[],
                discovery_time=time.time() - start_time,
                total_tools_analyzed=0
            )

    async def _score_tool_relevance(
        self,
        tool: Any,
        task_analysis: TaskAnalysis,
        agent_id: Optional[str] = None,
        context: Dict[str, Any] = None
    ) -> ToolSuggestion:
        """
        Score a tool's relevance to a task

        Scoring algorithm:
        - Keyword matching: 30% weight
        - Semantic similarity: 40% weight
        - Historical success: 30% weight

        Args:
            tool: Tool definition from registry
            task_analysis: Analyzed task
            agent_id: Optional agent ID for personalization
            context: Optional context for better scoring

        Returns:
            ToolSuggestion with relevance score
        """
        try:
            # Extract tool metadata
            tool_id = tool.tool_id
            tool_name = tool.name
            tool_category = tool.category
            tool_description = tool.description
            tool_tags = getattr(tool, 'tags', [])

            # 1. Keyword matching (30% weight)
            keyword_score = self._calculate_keyword_overlap(
                task_analysis.keywords,
                tool_tags + [tool_category.lower(), tool_name.lower()]
            )

            # 2. Semantic similarity (40% weight)
            semantic_score = await self._calculate_semantic_similarity(
                task_analysis.description,
                tool_description
            )

            # 3. Historical success rate (30% weight)
            success_score = await self._get_tool_success_rate(
                tool_id,
                task_analysis.intent
            )

            # Calculate weighted relevance score
            relevance_score = (
                keyword_score * 0.3 +
                semantic_score * 0.4 +
                success_score * 0.3
            )

            # Generate reason
            reason = self._generate_suggestion_reason(
                tool,
                task_analysis,
                keyword_score,
                semantic_score,
                success_score
            )

            # Get usage stats
            usage_count, overall_success_rate = await self._get_tool_usage_stats(tool_id)

            return ToolSuggestion(
                tool_id=tool_id,
                tool_name=tool_name,
                tool_category=tool_category,
                relevance_score=relevance_score,
                final_score=relevance_score,  # Will be adjusted in ranking
                reason=reason,
                keyword_match_score=keyword_score,
                semantic_similarity_score=semantic_score,
                historical_success_score=success_score,
                usage_count=usage_count,
                success_rate=overall_success_rate
            )

        except Exception as e:
            logger.error(f"❌ Failed to score tool {tool.tool_id}: {e}")
            # Return low-score suggestion on error
            return ToolSuggestion(
                tool_id=tool.tool_id,
                tool_name=tool.name,
                tool_category=tool.category,
                relevance_score=0.0,
                final_score=0.0,
                reason="Error scoring tool",
                keyword_match_score=0.0,
                semantic_similarity_score=0.0,
                historical_success_score=0.0,
                usage_count=0,
                success_rate=0.0
            )

    def _calculate_keyword_overlap(
        self,
        task_keywords: List[str],
        tool_keywords: List[str]
    ) -> float:
        """Calculate keyword overlap using Jaccard similarity"""
        if not task_keywords or not tool_keywords:
            return 0.0

        task_set = set(task_keywords)
        tool_set = set([k.lower() for k in tool_keywords])

        intersection = task_set.intersection(tool_set)
        union = task_set.union(tool_set)

        if not union:
            return 0.0

        return len(intersection) / len(union)

    async def _calculate_semantic_similarity(
        self,
        task_description: str,
        tool_description: str
    ) -> float:
        """
        Calculate semantic similarity between task and tool descriptions

        For now, uses simple word overlap. In future, can integrate with
        vector_search_service for embedding-based similarity.
        """
        try:
            # Simple word-based similarity for now
            task_words = set(re.findall(r'\b[a-z]{3,}\b', task_description.lower()))
            tool_words = set(re.findall(r'\b[a-z]{3,}\b', tool_description.lower()))

            # Remove stop words
            task_words = task_words - self.stop_words
            tool_words = tool_words - self.stop_words

            if not task_words or not tool_words:
                return 0.0

            intersection = task_words.intersection(tool_words)
            union = task_words.union(tool_words)

            return len(intersection) / len(union) if union else 0.0

        except Exception as e:
            logger.error(f"❌ Semantic similarity calculation failed: {e}")
            return 0.0

    async def _get_tool_success_rate(
        self,
        tool_id: str,
        task_intent: TaskIntent
    ) -> float:
        """Get historical success rate for a tool with specific intent"""
        try:
            # Query learning data
            learning_data = await self.db.find_one(
                self.learning_collection,
                {"tool_id": tool_id, "task_intent": task_intent.value}
            )

            if learning_data:
                return learning_data.get("success_rate", 0.5)

            # No data - return neutral score
            return 0.5

        except Exception as e:
            logger.error(f"❌ Failed to get success rate for {tool_id}: {e}")
            return 0.5

    async def _get_tool_usage_stats(self, tool_id: str) -> tuple:
        """Get overall usage statistics for a tool"""
        try:
            # Find all learning data for this tool across all intents
            learning_data_list = await self.db.find(
                self.learning_collection,
                {"tool_id": tool_id}
            )

            if not learning_data_list:
                return 0, 0.0

            # Aggregate manually
            total_uses = 0
            successful_uses = 0

            for data in learning_data_list:
                total_uses += data.get("total_uses", 0)
                successful_uses += data.get("successful_uses", 0)

            success_rate = successful_uses / total_uses if total_uses > 0 else 0.0
            return total_uses, success_rate

        except Exception as e:
            logger.error(f"❌ Failed to get usage stats for {tool_id}: {e}")
            return 0, 0.0

    def _generate_suggestion_reason(
        self,
        tool: Any,
        task_analysis: TaskAnalysis,
        keyword_score: float,
        semantic_score: float,
        success_score: float
    ) -> str:
        """Generate human-readable reason for tool suggestion"""
        reasons = []

        # Keyword match
        if keyword_score > 0.5:
            reasons.append("strong keyword match")
        elif keyword_score > 0.3:
            reasons.append("good keyword match")

        # Semantic similarity
        if semantic_score > 0.5:
            reasons.append("high semantic similarity")
        elif semantic_score > 0.3:
            reasons.append("relevant functionality")

        # Historical success
        if success_score > 0.7:
            reasons.append("excellent track record")
        elif success_score > 0.5:
            reasons.append("good success rate")

        # Task intent match
        if task_analysis.intent != TaskIntent.GENERAL:
            reasons.append(f"suitable for {task_analysis.intent.value} tasks")

        if reasons:
            return f"{tool.name}: {', '.join(reasons)}"
        else:
            return f"{tool.name}: may be relevant to this task"

    async def _rank_tools(
        self,
        tools: List[ToolSuggestion],
        agent_id: Optional[str] = None,
        context: Dict[str, Any] = None
    ) -> List[ToolSuggestion]:
        """
        Rank tools with contextual adjustments

        Applies boosts and penalties based on:
        - Agent's past successful usage
        - Recent success rate
        - Failure rate

        Args:
            tools: List of tool suggestions to rank
            agent_id: Optional agent ID for personalization
            context: Optional context for ranking

        Returns:
            Ranked list of tool suggestions
        """
        try:
            for tool in tools:
                score = tool.relevance_score

                # Boost for agent's past successful usage
                if agent_id:
                    agent_success = await self._get_agent_tool_success(
                        agent_id, tool.tool_id
                    )
                    if agent_success > 0.7:
                        score *= 1.2  # 20% boost
                    elif agent_success > 0.5:
                        score *= 1.1  # 10% boost

                # Boost for recent successful usage (last 7 days)
                recent_success = await self._get_recent_success_rate(
                    tool.tool_id, days=7
                )
                if recent_success > 0.8:
                    score *= 1.15  # 15% boost
                elif recent_success > 0.6:
                    score *= 1.05  # 5% boost

                # Penalty for high failure rate
                if tool.success_rate < 0.3:
                    score *= 0.7  # 30% penalty
                elif tool.success_rate < 0.5:
                    score *= 0.9  # 10% penalty

                # Cap at 1.0
                tool.final_score = min(score, 1.0)

            # Sort by final score
            tools.sort(key=lambda x: x.final_score, reverse=True)
            return tools

        except Exception as e:
            logger.error(f"❌ Tool ranking failed: {e}")
            # Return original list on error
            return tools

    async def _get_agent_tool_success(
        self,
        agent_id: str,
        tool_id: str
    ) -> float:
        """Get agent's success rate with specific tool"""
        try:
            # Query patterns for this agent and tool
            patterns = await self.db.find(
                self.patterns_collection,
                {
                    "agent_id": agent_id,
                    "tools_used": tool_id,
                    "success": True
                },
                limit=100
            )

            if not patterns:
                return 0.0

            successful = sum(1 for p in patterns if p.get("success", False))
            total = len(patterns)

            return successful / total if total > 0 else 0.0

        except Exception as e:
            logger.error(f"❌ Failed to get agent tool success: {e}")
            return 0.0

    async def _get_recent_success_rate(
        self,
        tool_id: str,
        days: int = 7
    ) -> float:
        """Get tool's success rate in recent days"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            # Query recent patterns
            patterns = await self.db.find(
                self.patterns_collection,
                {
                    "tools_used": tool_id,
                    "timestamp": {"$gte": cutoff_date}
                },
                limit=100
            )

            if not patterns:
                return 0.5  # Neutral if no recent data

            successful = sum(1 for p in patterns if p.get("success", False))
            total = len(patterns)

            return successful / total if total > 0 else 0.5

        except Exception as e:
            logger.error(f"❌ Failed to get recent success rate: {e}")
            return 0.5

    async def learn_from_execution(
        self,
        task_description: str,
        tools_used: List[str],
        tools_suggested: List[str],
        success: bool,
        execution_time: float = 0.0,
        agent_id: Optional[str] = None,
        agent_type: Optional[str] = None
    ):
        """
        Learn from execution to improve future suggestions

        This method stores execution patterns and updates learning data
        to improve tool discovery accuracy over time.

        Args:
            task_description: Original task description
            tools_used: List of tool IDs actually used
            tools_suggested: List of tool IDs that were suggested
            success: Whether execution was successful
            execution_time: Execution time in seconds
            agent_id: Optional agent ID
            agent_type: Optional agent type
        """
        try:
            # Analyze the task
            task_analysis = await self.analyze_task(task_description)

            # Calculate suggestion accuracy
            if tools_suggested:
                tools_used_set = set(tools_used)
                tools_suggested_set = set(tools_suggested)
                correct_suggestions = tools_used_set.intersection(tools_suggested_set)
                accuracy = len(correct_suggestions) / len(tools_suggested_set)
            else:
                accuracy = 0.0

            # Store usage pattern
            pattern = ToolUsagePattern(
                pattern_id=f"pattern_{uuid.uuid4().hex[:12]}",
                task_description=task_description,
                task_intent=task_analysis.intent,
                tools_used=tools_used,
                tools_suggested=tools_suggested,
                success=success,
                execution_time=execution_time,
                agent_id=agent_id,
                agent_type=agent_type,
                suggestion_accuracy=accuracy,
                tools_used_count=len(tools_used),
                tools_suggested_count=len(tools_suggested)
            )

            await self.db.insert_one(
                self.patterns_collection,
                pattern.model_dump()
            )

            # Update learning data for each tool used
            for tool_id in tools_used:
                await self._update_tool_learning_data(
                    tool_id,
                    task_analysis.intent,
                    success,
                    execution_time
                )

            logger.info(f"✅ Learned from execution: {len(tools_used)} tools, accuracy={accuracy:.2f}")

        except Exception as e:
            logger.error(f"❌ Failed to learn from execution: {e}")
            import traceback
            logger.error(traceback.format_exc())

    async def _update_tool_learning_data(
        self,
        tool_id: str,
        task_intent: TaskIntent,
        success: bool,
        execution_time: float
    ):
        """Update learning data for a specific tool and intent"""
        try:
            # Find existing learning data
            existing = await self.db.find_one(
                self.learning_collection,
                {"tool_id": tool_id, "task_intent": task_intent.value}
            )

            if existing:
                # Update existing data
                total_uses = existing.get("total_uses", 0) + 1
                successful_uses = existing.get("successful_uses", 0) + (1 if success else 0)
                failed_uses = existing.get("failed_uses", 0) + (0 if success else 1)
                success_rate = successful_uses / total_uses if total_uses > 0 else 0.0

                # Update execution time stats
                avg_time = existing.get("avg_execution_time", 0.0)
                new_avg_time = (avg_time * (total_uses - 1) + execution_time) / total_uses

                min_time = min(existing.get("min_execution_time", execution_time), execution_time)
                max_time = max(existing.get("max_execution_time", execution_time), execution_time)

                await self.db.update_one(
                    self.learning_collection,
                    {"tool_id": tool_id, "task_intent": task_intent.value},
                    {
                        "$set": {
                            "total_uses": total_uses,
                            "successful_uses": successful_uses,
                            "failed_uses": failed_uses,
                            "success_rate": success_rate,
                            "avg_execution_time": new_avg_time,
                            "min_execution_time": min_time,
                            "max_execution_time": max_time,
                            "last_used": datetime.utcnow()
                        }
                    }
                )
            else:
                # Create new learning data
                learning_data = ToolLearningData(
                    tool_id=tool_id,
                    task_intent=task_intent,
                    total_uses=1,
                    successful_uses=1 if success else 0,
                    failed_uses=0 if success else 1,
                    success_rate=1.0 if success else 0.0,
                    avg_execution_time=execution_time,
                    min_execution_time=execution_time,
                    max_execution_time=execution_time,
                    last_used=datetime.utcnow(),
                    first_used=datetime.utcnow()
                )

                await self.db.insert_one(
                    self.learning_collection,
                    learning_data.model_dump()
                )

        except Exception as e:
            logger.error(f"❌ Failed to update tool learning data: {e}")

    async def _update_discovery_stats(
        self,
        tools_analyzed: int,
        suggestions_made: int,
        discovery_time: float
    ):
        """Update overall discovery statistics"""
        try:
            stats = await self.db.find_one(self.stats_collection, {})

            if stats:
                # Update existing stats
                total_discoveries = stats.get("total_discoveries", 0) + 1
                total_suggestions = stats.get("total_suggestions", 0) + suggestions_made
                avg_suggestions = total_suggestions / total_discoveries

                # Update average discovery time
                avg_time = stats.get("avg_discovery_time", 0.0)
                new_avg_time = (avg_time * (total_discoveries - 1) + discovery_time) / total_discoveries

                await self.db.update_one(
                    self.stats_collection,
                    {},
                    {
                        "$set": {
                            "total_discoveries": total_discoveries,
                            "total_suggestions": total_suggestions,
                            "avg_suggestions_per_request": avg_suggestions,
                            "avg_discovery_time": new_avg_time
                        }
                    }
                )
            else:
                # Create new stats
                stats = ToolDiscoveryStats(
                    total_discoveries=1,
                    total_suggestions=suggestions_made,
                    avg_suggestions_per_request=suggestions_made,
                    avg_discovery_time=discovery_time
                )

                await self.db.insert_one(
                    self.stats_collection,
                    stats.model_dump()
                )

        except Exception as e:
            logger.error(f"❌ Failed to update discovery stats: {e}")

    async def get_stats(self) -> ToolDiscoveryStats:
        """Get overall tool discovery statistics"""
        try:
            stats = await self.db.find_one(self.stats_collection, {})

            if stats:
                return ToolDiscoveryStats(**stats)
            else:
                return ToolDiscoveryStats()

        except Exception as e:
            logger.error(f"❌ Failed to get stats: {e}")
            return ToolDiscoveryStats()


# Global instance
tool_discovery_service = None


def get_tool_discovery_service(db: Database) -> ToolDiscoveryService:
    """Get or create tool discovery service instance"""
    global tool_discovery_service
    if tool_discovery_service is None:
        tool_discovery_service = ToolDiscoveryService(db)
    return tool_discovery_service

