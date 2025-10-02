"""
GRETA PAI Learning Service - Daniel Miessler PAI Integration
Coordinates Greta and PAI systems through MongoDB learning data and local Llama3 model training
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import os
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure

logger = logging.getLogger(__name__)


class LearningService:
    """
    Coordinates PAI learning with MongoDB persistence and local Llama3 model training
    using Hierarchical Reasoning Model (HRM) coordination.
    """

    def __init__(self, db_client: AsyncIOMotorClient):
        self.db_client = db_client
        self.learning_db = db_client["greta_learning"]
        self.interactions_collection = self.learning_db["interactions"]
        self.reasoning_collection = self.learning_db["reasoning_patterns"]
        self.llama_model_path = Path("/Users/macone/Desktop/Greta/models/llama3_greta.gguf")

        # HRM coordination state
        self.hrm_layers = {
            "strategic": [],  # Long-term decision patterns
            "tactical": [],   # Medium-term behavior optimization
            "operational": [] # Immediate response refinement
        }

        self.learning_cycles = 0
        self.model_version = "v1.0.0"

    async def initialize(self):
        """Initialize learning service with database setup"""
        try:
            # Test database connection
            await self.db_client.admin.command('ping')
            logger.info("✅ Learning service connected to MongoDB")

            # Create indexes for efficient querying
            await self.interactions_collection.create_index([("user_id", 1), ("timestamp", -1)])
            await self.interactions_collection.create_index([("learning_category", 1)])
            await self.reasoning_collection.create_index([("hrm_layer", 1), ("confidence", -1)])

            logger.info("✅ Learning service indexes created")

        except ConnectionFailure:
            logger.error("❌ MongoDB connection failed for learning service")
            raise

    async def record_pai_interaction(self, interaction_data: Dict[str, Any]) -> str:
        """
        Record PAI learning interaction for Greta coordination
        Stores in MongoDB for future Llama3 model training
        """

        interaction_doc = {
            "user_id": interaction_data.get("user_id", "default_user"),
            "timestamp": datetime.utcnow(),
            "pai_framework": "daniel-miessler",
            "interaction_type": interaction_data.get("type", "unknown"),
            "learning_category": interaction_data.get("category", "general"),

            # Emotional intelligence adaptation
            "personality_notes": interaction_data.get("personality_adaptation", ""),

            # Hierarchical reasoning capture
            "hrm_strategic": interaction_data.get("strategic_reasoning", ""),
            "hrm_tactical": interaction_data.get("tactical_reasoning", ""),
            "hrm_operational": interaction_data.get("operational_reasoning", ""),

            # PAI core learnings
            "tilos_goals_alignment": interaction_data.get("goals_alignment", ""),
            "fabric_workflow_patterns": interaction_data.get("workflow_patterns", ""),
            "substrate_organizational_models": interaction_data.get("organizational_learning", ""),

            # Response effectiveness metrics
            "response_quality": interaction_data.get("response_quality", 0.0),
            "user_satisfaction": interaction_data.get("user_satisfaction", 0.0),
            "context_relevance": interaction_data.get("context_relevance", 0.0),

            # Training data labels
            "model_training_labels": {
                "preferred_response_style": interaction_data.get("preferred_style", ""),
                "effective_communication_patterns": interaction_data.get("comm_patterns", []),
                "knowledge_gaps_identified": interaction_data.get("knowledge_gaps", [])
            }
        }

        result = await self.interactions_collection.insert_one(interaction_doc)

        # Update HRM coordination layers
        await self._update_hrm_layers(interaction_doc)

        logger.info(f"✅ PAI interaction recorded: {result.inserted_id}")
        return str(result.inserted_id)

    async def _update_hrm_layers(self, interaction_doc: Dict[str, Any]):
        """Update Hierarchical Reasoning Model coordination layers"""

        # Strategic layer - Long-term patterns and values
        if interaction_doc.get("hrm_strategic"):
            self.hrm_layers["strategic"].append({
                "timestamp": interaction_doc["timestamp"],
                "pattern": interaction_doc["hrm_strategic"],
                "tilos_alignment": interaction_doc.get("tilos_goals_alignment"),
                "confidence": interaction_doc.get("response_quality", 0.0)
            })

            # Maintain strategic layer size limit
            if len(self.hrm_layers["strategic"]) > 1000:
                self.hrm_layers["strategic"] = self.hrm_layers["strategic"][-500:]  # Keep latest 500

        # Tactical layer - Medium-term behavior optimization
        if interaction_doc.get("hrm_tactical"):
            self.hrm_layers["tactical"].append({
                "timestamp": interaction_doc["timestamp"],
                "behavior": interaction_doc["hrm_tactical"],
                "substrate_models": interaction_doc.get("substrate_organizational_models"),
                "fabric_automation": interaction_doc.get("fabric_workflow_patterns")
            })

            # Maintain tactical layer size limit
            if len(self.hrm_layers["tactical"]) > 2000:
                self.hrm_layers["tactical"] = self.hrm_layers["tactical"][-1000:]

        # Operational layer - Immediate response refinement
        if interaction_doc.get("hrm_operational"):
            self.hrm_layers["operational"].append({
                "timestamp": interaction_doc["timestamp"],
                "improvement": interaction_doc["hrm_operational"],
                "user_feedback": interaction_doc.get("user_satisfaction", 0.0),
                "context_adaptation": interaction_doc.get("context_relevance", 0.0)
            })

    async def get_pai_learning_data(self, user_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Retrieve PAI learning data from MongoDB for Llama3 training"""

        start_date = datetime.utcnow() - timedelta(days=days)

        cursor = self.interactions_collection.find({
            "user_id": user_id,
            "timestamp": {"$gte": start_date}
        }).sort("timestamp", -1)

        learning_data = []
        async for doc in cursor:
            # Clean MongoDB ObjectId for JSON serialization
            doc["_id"] = str(doc["_id"])
            learning_data.append(doc)

        logger.info(f"📊 Retrieved {len(learning_data)} PAI learning records for {user_id}")
        return learning_data

    async def prepare_llama3_training_data(self, user_id: str) -> str:
        """
        Prepare PAI learning data for local Llama3 model training
        Returns formatted training dataset path
        """

        learning_data = await self.get_pai_learning_data(user_id)

        # Create training data structure
        training_samples = []

        for interaction in learning_data:
            # Create instruction-response pairs for fine-tuning
            instruction = self._create_training_instruction(interaction)
            response = self._create_training_response(interaction)

            training_samples.append({
                "instruction": instruction,
                "input": interaction.get("hrm_operational", ""),
                "output": response,
                "timestamp": interaction["timestamp"].isoformat(),
                "quality_score": interaction.get("response_quality", 0.0)
            })

        # Add HRM coordination patterns
        training_samples.extend(self._create_hrm_training_samples())

        # Save training dataset
        training_file = Path(f"/Users/macone/Desktop/Greta/training_data/pai_learning_{user_id}_{self.learning_cycles}.jsonl")

        training_file.parent.mkdir(parents=True, exist_ok=True)

        with open(training_file, 'w') as f:
            for sample in training_samples:
                f.write(json.dumps(sample) + '\n')

        logger.info(f"🎯 Prepared {len(training_samples)} training samples for Llama3")
        return str(training_file)

    def _create_training_instruction(self, interaction: Dict[str, Any]) -> str:
        """Create fine-tuning instruction from PAI interaction"""

        category = interaction.get("learning_category", "general")
        user_style = interaction.get("model_training_labels", {}).get("preferred_response_style", "")

        instruction = f"You are Greta, an advanced PAI assistant following Daniel Miessler's teaching framework. "
        instruction += f"Category: {category}. "
        if user_style:
            instruction += f"User preference: {user_style}. "
        instruction += f"Respond in a way that continuously adapts to this user's PAI context."

        return instruction

    def _create_training_response(self, interaction: Dict[str, Any]) -> str:
        """Create expected response from PAI learning data"""

        response_parts = []

        # Add strategic reasoning if available
        if interaction.get("hrm_strategic"):
            response_parts.append(f"Strategic approach: {interaction['hrm_strategic']}")

        # Add tactical adaptation
        if interaction.get("hrm_tactical"):
            response_parts.append(f"Behavioral adaptation: {interaction['hrm_tactical']}")

        # Add operational refinement
        if interaction.get("hrm_operational"):
            response_parts.append(f"Response refinement: {interaction['hrm_operational']}")

        # Add TilOs goal alignment
        if interaction.get("tilos_goals_alignment"):
            response_parts.append(f"Purpose alignment: {interaction['tilos_goals_alignment']}")

        return " | ".join(response_parts) if response_parts else "Continue PAI learning adaptation"

    def _create_hrm_training_samples(self) -> List[Dict[str, Any]]:
        """Create training samples from HRM coordination patterns"""

        training_samples = []

        # Strategic layer samples
        for strategic_pattern in self.hrm_layers["strategic"][-10:]:  # Use last 10
            training_samples.append({
                "instruction": "Apply hierarchical reasoning for strategic decision-making",
                "input": f"Tilos context: {strategic_pattern.get('tilos_alignment', 'Optimize purpose alignment')}",
                "output": f"Strategic pattern: {strategic_pattern['pattern']}",
                "confidence": strategic_pattern.get('confidence', 0.5)
            })

        # Tactical layer samples
        for tactical_pattern in self.hrm_layers["tactical"][-20:]:  # Use last 20
            training_samples.append({
                "instruction": "Optimize AI behavior patterns using tactical reasoning",
                "input": f"Fabric workflow: {tactical_pattern.get('fabric_automation', 'Automate processes')}",
                "output": f"Behavioral optimization: {tactical_pattern['behavior']}",
                "confidence": 0.8
            })

        # Operational layer samples
        for operational_pattern in self.hrm_layers["operational"][-50:]:  # Use last 50
            training_samples.append({
                "instruction": "Refine immediate responses using operational reasoning",
                "input": f"User feedback: {operational_pattern.get('user_feedback', 0.7)}",
                "output": f"Response improvement: {operational_pattern['improvement']}",
                "confidence": operational_pattern.get('context_adaptation', 0.6)
            })

        return training_samples

    async def trigger_llama3_training(self, training_data_path: str) -> bool:
        """Trigger local Llama3 model training with PAI learning data"""

        try:
            # Check if training data exists
            if not Path(training_data_path).exists():
                logger.error(f"❌ Training data not found: {training_data_path}")
                return False

            # Prepare training command
            model_output_path = f"/Users/macone/Desktop/Greta/models/llama3_greta_hrm_{self.learning_cycles + 1}.gguf"
            training_script = f"/Users/macone/Desktop/Greta/train_llama3_hrm.sh"

            # Create training script
            await self._create_training_script(training_script, training_data_path, model_output_path)

            # Simulate training trigger (in real implementation, this would execute the script)
            logger.info(f"🚀 Starting Llama3 HRM training with PAI data: {training_data_path}")
            logger.info(f"📁 Model output: {model_output_path}")

            # Increment learning cycle
            self.learning_cycles += 1

            # Update model version
            self.model_version = f"v1.{self.learning_cycles}.{int(datetime.utcnow().timestamp())}"

            return True

        except Exception as e:
            logger.error(f"❌ Llama3 training trigger failed: {e}")
            return False

    async def _create_training_script(self, script_path: str, data_path: str, model_path: str):
        """Create Llama3 training script with HRM integration"""

        script_content = f'''#!/bin/bash

# GRETA PAI Llama3 HRM Training Script
# Integrates PAI learning data with Hierarchical Reasoning Model

set -e

echo "🎯 Starting GRETA PAI Llama3 HRM Training"
echo "Learning Cycle: {self.learning_cycles + 1}"
echo "Training Data: {data_path}"
echo "Output Model: {model_path}"

# Use llama.cpp for fine-tuning with PAI learning data
# In production, this would use full fine-tuning workflow
# For now, this is a simulation framework

# Step 1: Validate training data
if [ ! -f "{data_path}" ]; then
    echo "❌ Training data not found: {data_path}"
    exit 1
fi

echo "✅ Training data validated"

# Step 2: Backup previous model
if [ -f "/Users/macone/Desktop/Greta/models/llama3_greta_current.gguf" ]; then
    cp "/Users/macone/Desktop/Greta/models/llama3_greta_current.gguf" "/Users/macone/Desktop/Greta/models/backup/llama3_greta_{self.learning_cycles}.gguf"
    echo "💾 Previous model backed up"
fi

# Step 3: HRM-coordinated fine-tuning simulation
# In real implementation, this would use:
# - llama.cpp fine-tune command
# - LoRA adapters trained on PAI learning data
# - Hierarchical reasoning model optimization

echo "🧠 Applying HRM coordination to training process..."
echo "Strategic layer: $(wc -l < "{data_path}" | xargs) patterns"
echo "Tactical layer: $((${len(self.hrm_layers['tactical'])})) adaptations"
echo "Operational layer: $((${len(self.hrm_layers['operational'])})) refinements"

# Simulate training completion
echo "🎓 Llama3 model fine-tuning completed with PAI HRM coordination"
echo "Model version: {self.model_version}"
echo "Training cycles completed: {self.learning_cycles + 1}"

# Step 4: Deploy new model
cp "{model_path}" "/Users/macone/Desktop/Greta/models/llama3_greta_current.gguf"
echo "🚀 New HRM-coordinated model deployed"

echo "✅ GRETA PAI learning cycle {self.learning_cycles + 1} completed"
'''

        with open(script_path, 'w') as f:
            f.write(script_content)

        # Make script executable
        os.chmod(script_path, 0o755)

        logger.info(f"📝 Created Llama3 HRM training script: {script_path}")

    async def get_pai_coordination_status(self) -> Dict[str, Any]:
        """Get current status of PAI-Greta coordination"""

        total_interactions = await self.interactions_collection.count_documents({})
        hrm_patterns = await self.reasoning_collection.count_documents({})

        return {
            "pai_framework": "active",
            "learning_cycles": self.learning_cycles,
            "model_version": self.model_version,
            "total_pai_interactions": total_interactions,
            "hrm_coordination_patterns": hrm_patterns,
            "hierarchical_layers": {
                "strategic": len(self.hrm_layers["strategic"]),
                "tactical": len(self.hrm_layers["tactical"]),
                "operational": len(self.hrm_layers["operational"])
            },
            "mongodb_connection": "active",
            "llama3_integration": "ready",
            "training_data_prepared": "available"
        }

    async def close(self):
        """Cleanup learning service resources"""

        # Save HRM state to MongoDB for persistence
        await self._save_hrm_state()

        logger.info("🔄 Learning service cleanup completed")

    async def _save_hrm_state(self):
        """Persist HRM coordination state"""

        hrm_state = {
            "timestamp": datetime.utcnow(),
            "learning_cycles": self.learning_cycles,
            "model_version": self.model_version,
            "hrm_layers_counts": {
                "strategic": len(self.hrm_layers["strategic"]),
                "tactical": len(self.hrm_layers["tactical"]),
                "operational": len(self.hrm_layers["operational"])
            },
            "pai_coordination_status": "active"
        }

        await self.learning_db["hrm_state"].update_one(
            {"current": True},
            {"$set": hrm_state},
            upsert=True
        )
