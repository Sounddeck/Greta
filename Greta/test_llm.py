"""
Test script for LLM Service
Week 1 - Task 5: Test Basic Inference
"""

import asyncio
import sys
sys.path.insert(0, '/Users/macone/Desktop/Greta/Greta')

from backend.services.llm_service import llm_service
from loguru import logger


async def test_llm():
    """Test LLM service functionality"""
    
    logger.info("=" * 60)
    logger.info("LLM Service Test - Week 1")
    logger.info("=" * 60)
    
    # Test 1: List available models
    logger.info("\n📋 Test 1: List available models")
    models = llm_service.list_models()
    logger.info(f"Found {len(models)} models: {models}")
    
    if not models:
        logger.error("❌ No models found! Please download a model first.")
        return False
    
    # Use first available model
    model_name = models[0]
    logger.success(f"✅ Will use model: {model_name}")
    
    # Test 2: Get model info
    logger.info(f"\n📊 Test 2: Get model info for {model_name}")
    model_info = llm_service.get_model_info(model_name)
    if model_info:
        logger.info(f"Model: {model_info['name']}")
        logger.info(f"Size: {model_info['size_mb']} MB")
        logger.info(f"Path: {model_info['path']}")
        logger.success("✅ Model info retrieved")
    else:
        logger.error("❌ Failed to get model info")
        return False
    
    # Test 3: Load model
    logger.info(f"\n🔄 Test 3: Load model {model_name}")
    logger.info("This may take 10-30 seconds...")
    success = await llm_service.load_model(model_name, n_ctx=2048, n_gpu_layers=35)
    
    if not success:
        logger.error("❌ Failed to load model")
        return False
    
    logger.success(f"✅ Model {model_name} loaded successfully")
    
    # Test 4: Generate simple response
    logger.info("\n🤖 Test 4: Generate simple response")
    prompt = "Hello! Please introduce yourself in one sentence."
    logger.info(f"Prompt: {prompt}")
    
    try:
        response = await llm_service.generate(
            prompt=prompt,
            max_tokens=100,
            temperature=0.7
        )
        
        logger.success("✅ Generation successful!")
        logger.info(f"\nResponse:\n{response}\n")
        
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        return False
    
    # Test 5: Generate with different parameters
    logger.info("\n🎯 Test 5: Generate with different parameters")
    prompt2 = "What is 2+2? Answer in one word."
    logger.info(f"Prompt: {prompt2}")
    
    try:
        response2 = await llm_service.generate(
            prompt=prompt2,
            max_tokens=10,
            temperature=0.1  # Lower temperature for more deterministic output
        )
        
        logger.success("✅ Generation successful!")
        logger.info(f"\nResponse:\n{response2}\n")
        
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        return False
    
    # Test 6: List loaded models
    logger.info("\n📝 Test 6: List loaded models")
    loaded = llm_service.list_loaded_models()
    logger.info(f"Loaded models: {loaded}")
    logger.success("✅ Loaded models listed")
    
    # Test 7: Unload model
    logger.info(f"\n🔄 Test 7: Unload model {model_name}")
    success = await llm_service.unload_model(model_name)
    
    if success:
        logger.success(f"✅ Model {model_name} unloaded")
    else:
        logger.error("❌ Failed to unload model")
        return False
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.success("✅ ALL TESTS PASSED!")
    logger.info("=" * 60)
    logger.info("\nWeek 1 Task 5 Complete: Basic inference working!")
    
    return True


if __name__ == "__main__":
    result = asyncio.run(test_llm())
    sys.exit(0 if result else 1)
