"""
Conversation Collector for ART Training
Collects and prepares conversation data for agent training
"""
import logging
from typing import List, Optional
from datetime import datetime, timedelta

from models.art_training_models import ConversationTurn
from database import get_database

logger = logging.getLogger(__name__)


class ConversationCollector:
    """Collects conversation data for training"""
    
    def __init__(self):
        """Initialize conversation collector"""
        self.db = get_database()
        self.conversations_collection = "agent_conversations"
        logger.info("✅ Conversation Collector initialized")
    
    async def collect_conversations(
        self,
        agent_id: str,
        days_back: int = 30,
        min_rating: Optional[float] = None,
        max_conversations: Optional[int] = None
    ) -> List[ConversationTurn]:
        """
        Collect conversations for an agent
        
        Args:
            agent_id: Agent ID to collect for
            days_back: Number of days to look back
            min_rating: Minimum user rating to include
            max_conversations: Maximum number of conversations to return
            
        Returns:
            List[ConversationTurn]: Collected conversations
        """
        try:
            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days_back)
            
            # Build query
            query = {
                "agent_id": agent_id,
                "timestamp": {"$gte": start_date, "$lte": end_date}
            }
            
            if min_rating is not None:
                query["user_rating"] = {"$gte": min_rating}
            
            # Fetch conversations
            cursor = self.db.find(
                self.conversations_collection,
                query,
                sort=[("timestamp", -1)]
            )
            
            conversations = []
            async for doc in cursor:
                try:
                    conversation = ConversationTurn(**doc)
                    conversations.append(conversation)
                    
                    if max_conversations and len(conversations) >= max_conversations:
                        break
                        
                except Exception as e:
                    logger.warning(f"Failed to parse conversation: {e}")
                    continue
            
            logger.info(f"✅ Collected {len(conversations)} conversations for agent {agent_id}")
            return conversations
            
        except Exception as e:
            logger.error(f"❌ Failed to collect conversations: {e}")
            return []
    
    async def store_conversation(self, conversation: ConversationTurn) -> bool:
        """
        Store a conversation turn for future training
        
        Args:
            conversation: Conversation turn to store
            
        Returns:
            bool: Success status
        """
        try:
            await self.db.insert_one(
                self.conversations_collection,
                conversation.dict()
            )
            return True
        except Exception as e:
            logger.error(f"❌ Failed to store conversation: {e}")
            return False

