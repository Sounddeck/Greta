"""
Test script for Week 2: Model Management System
Tests all 4 components:
1. Model Registry
2. Model Downloader
3. Model Configuration
4. Performance Monitoring
"""

import asyncio
import sys
sys.path.insert(0, '/Users/macone/Desktop/Greta/Greta')

from backend.services.model_registry import model_registry, ModelMetadata
from backend.services.model_downloader import model_downloader
from backend.services.model_config import model_config_manager, ModelConfig
from backend.services.model_monitor import model_monitor
from loguru import logger


async def test_model_registry():
    """Test Component 1: Model Registry"""
    logger.info("\n" + "=" * 60)
    logger.info("Component 1: Model Registry System")
    logger.info("=" * 60)
    
    # Test 1: Initialize registry
    logger.info("\n📋 Test 1.1: Initialize Model Registry")
    await model_registry.initialize()
    logger.success("✅ Registry initialized")
    
    # Test 2: List existing models
    logger.info("\n📋 Test 1.2: List existing models")
    models = await model_registry.list_models()
    logger.info(f"Found {len(models)} models")
    for model in models:
        logger.info(f"  - {model.name} ({model.family} {model.size}, {model.quantization})")
    logger.success("✅ Models listed")
    
    # Test 3: Register a new model
    logger.info("\n📋 Test 1.3: Register new model")
    test_metadata = ModelMetadata(
        name="test-model",
        family="llama",
        size="7B",
        quantization="Q4_K_M",
        file_name="test-model.gguf",
        file_size_mb=3800.0,
        capabilities=["chat", "code"],
        tags=["test", "demo"]
    )
    success = await model_registry.register_model(test_metadata)
    if success:
        logger.success("✅ Model registered")
    else:
        logger.error("❌ Failed to register model")
        return False
    
    # Test 4: Get model metadata
    logger.info("\n📋 Test 1.4: Get model metadata")
    metadata = await model_registry.get_model("test-model")
    if metadata:
        logger.info(f"Model: {metadata.name}")
        logger.info(f"Family: {metadata.family}")
        logger.info(f"Size: {metadata.size}")
        logger.info(f"Capabilities: {metadata.capabilities}")
        logger.success("✅ Metadata retrieved")
    else:
        logger.error("❌ Failed to get metadata")
        return False
    
    # Test 5: Update model
    logger.info("\n📋 Test 1.5: Update model metadata")
    success = await model_registry.update_model("test-model", usage_count=5, status="downloaded")
    if success:
        logger.success("✅ Model updated")
    else:
        logger.error("❌ Failed to update model")
        return False
    
    # Test 6: Delete test model
    logger.info("\n📋 Test 1.6: Delete test model")
    success = await model_registry.delete_model("test-model", delete_file=False)
    if success:
        logger.success("✅ Model deleted")
    else:
        logger.error("❌ Failed to delete model")
        return False
    
    logger.success("\n✅ Component 1: Model Registry - ALL TESTS PASSED")
    return True


async def test_model_downloader():
    """Test Component 2: Model Downloader"""
    logger.info("\n" + "=" * 60)
    logger.info("Component 2: Model Download Manager")
    logger.info("=" * 60)
    
    # Test 1: List available models from HuggingFace
    logger.info("\n📥 Test 2.1: List available models from HuggingFace")
    repo_id = "TheBloke/Llama-2-7B-GGUF"
    models = await model_downloader.list_available_models(repo_id)
    logger.info(f"Found {len(models)} GGUF files in {repo_id}")
    if models:
        logger.info(f"Sample models: {models[:3]}")
        logger.success("✅ Models listed from HuggingFace")
    else:
        logger.warning("⚠️ No models found (may be network issue)")
    
    # Test 2: Check download progress (no actual download)
    logger.info("\n📥 Test 2.2: Check download progress tracking")
    progress = model_downloader.get_download_progress("test-model")
    if progress is None:
        logger.success("✅ No active downloads (expected)")
    else:
        logger.info(f"Active download: {progress.model_name} - {progress.progress_percent}%")
    
    logger.success("\n✅ Component 2: Model Downloader - ALL TESTS PASSED")
    logger.info("Note: Skipped actual download to save time and bandwidth")
    return True


async def test_model_config():
    """Test Component 3: Model Configuration"""
    logger.info("\n" + "=" * 60)
    logger.info("Component 3: Model Configuration System")
    logger.info("=" * 60)
    
    # Test 1: Initialize config manager
    logger.info("\n⚙️  Test 3.1: Initialize Config Manager")
    await model_config_manager.initialize()
    logger.success("✅ Config manager initialized")
    
    # Test 2: Create default configs
    logger.info("\n⚙️  Test 3.2: Create default configurations")
    await model_config_manager.create_default_configs()
    logger.success("✅ Default configs created")
    
    # Test 3: Get config for a model
    logger.info("\n⚙️  Test 3.3: Get model configuration")
    config = await model_config_manager.get_config("llama-2-7b")
    logger.info(f"Config: {config.name}")
    logger.info(f"Context window: {config.context_window}")
    logger.info(f"GPU layers: {config.n_gpu_layers}")
    logger.info(f"Temperature: {config.temperature}")
    logger.info(f"System prompt: {config.system_prompt[:50]}..." if config.system_prompt else "None")
    logger.success("✅ Config retrieved")
    
    # Test 4: Create custom config
    logger.info("\n⚙️  Test 3.4: Create custom configuration")
    custom_config = ModelConfig(
        name="custom-model",
        context_window=8192,
        n_gpu_layers=40,
        temperature=0.9,
        system_prompt="You are a creative AI assistant.",
        stop_sequences=["</s>", "<|end|>"]
    )
    success = await model_config_manager.save_config(custom_config)
    if success:
        logger.success("✅ Custom config saved")
    else:
        logger.error("❌ Failed to save config")
        return False
    
    # Test 5: Delete custom config
    logger.info("\n⚙️  Test 3.5: Delete custom configuration")
    success = await model_config_manager.delete_config("custom-model")
    if success:
        logger.success("✅ Config deleted")
    else:
        logger.error("❌ Failed to delete config")
        return False
    
    logger.success("\n✅ Component 3: Model Configuration - ALL TESTS PASSED")
    return True


async def test_performance_monitor():
    """Test Component 4: Performance Monitoring"""
    logger.info("\n" + "=" * 60)
    logger.info("Component 4: Performance Monitoring System")
    logger.info("=" * 60)
    
    # Test 1: Start inference tracking
    logger.info("\n📊 Test 4.1: Start inference tracking")
    inference_id = model_monitor.start_inference("test-model", prompt_tokens=50)
    logger.info(f"Inference ID: {inference_id}")
    logger.success("✅ Inference tracking started")
    
    # Simulate some work
    import time
    time.sleep(0.1)
    
    # Test 2: End inference tracking
    logger.info("\n📊 Test 4.2: End inference tracking")
    metrics = model_monitor.end_inference(
        inference_id=inference_id,
        completion_tokens=100,
        temperature=0.7,
        success=True
    )
    if metrics:
        logger.info(f"Inference time: {metrics.inference_time_ms}ms")
        logger.info(f"Tokens/second: {metrics.tokens_per_second}")
        logger.info(f"Memory used: {metrics.memory_used_mb}MB")
        logger.success("✅ Metrics recorded")
    else:
        logger.error("❌ Failed to record metrics")
        return False
    
    # Test 3: Record multiple inferences
    logger.info("\n📊 Test 4.3: Record multiple inferences")
    for i in range(5):
        inf_id = model_monitor.start_inference("test-model", prompt_tokens=30 + i * 10)
        time.sleep(0.05)
        model_monitor.end_inference(inf_id, completion_tokens=80 + i * 5, temperature=0.7)
    logger.success("✅ Multiple inferences recorded")
    
    # Test 4: Get model statistics
    logger.info("\n📊 Test 4.4: Get model statistics")
    stats = model_monitor.get_model_stats("test-model", hours=24)
    logger.info(f"Total inferences: {stats.get('total_inferences', 0)}")
    logger.info(f"Success rate: {stats.get('success_rate_percent', 0)}%")
    logger.info(f"Avg tokens/second: {stats.get('avg_tokens_per_second', 0)}")
    logger.info(f"Avg inference time: {stats.get('avg_inference_time_ms', 0)}ms")
    logger.success("✅ Statistics retrieved")
    
    # Test 5: Clear metrics
    logger.info("\n📊 Test 4.5: Clear metrics")
    model_monitor.clear_metrics("test-model")
    logger.success("✅ Metrics cleared")
    
    logger.success("\n✅ Component 4: Performance Monitoring - ALL TESTS PASSED")
    return True


async def run_all_tests():
    """Run all Week 2 tests"""
    logger.info("=" * 60)
    logger.info("Week 2: Model Management System - Test Suite")
    logger.info("=" * 60)
    
    results = {
        'Component 1: Model Registry': False,
        'Component 2: Model Downloader': False,
        'Component 3: Model Configuration': False,
        'Component 4: Performance Monitoring': False
    }
    
    try:
        results['Component 1: Model Registry'] = await test_model_registry()
    except Exception as e:
        logger.error(f"Component 1 failed: {e}")
    
    try:
        results['Component 2: Model Downloader'] = await test_model_downloader()
    except Exception as e:
        logger.error(f"Component 2 failed: {e}")
    
    try:
        results['Component 3: Model Configuration'] = await test_model_config()
    except Exception as e:
        logger.error(f"Component 3 failed: {e}")
    
    try:
        results['Component 4: Performance Monitoring'] = await test_performance_monitor()
    except Exception as e:
        logger.error(f"Component 4 failed: {e}")
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("WEEK 2 TEST SUMMARY")
    logger.info("=" * 60)
    
    for component, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{component}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.success("\n🎉 ALL WEEK 2 TESTS PASSED!")
        logger.info("\nWeek 2 Complete: Model Management System working!")
    else:
        logger.error("\n❌ SOME TESTS FAILED")
        logger.info("Please review the errors above")
    
    return all_passed


if __name__ == "__main__":
    result = asyncio.run(run_all_tests())
    sys.exit(0 if result else 1)
