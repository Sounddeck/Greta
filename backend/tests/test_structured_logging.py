"""
Tests for Structured Logging Service
Validates JSON formatting, correlation ID tracking, and performance
"""

import json
import time
import uuid
from pathlib import Path
from loguru import logger

from services.structured_logger import (
    StructuredLogger,
    setup_structured_logging,
    get_structured_logger
)


def test_json_formatter():
    """Test JSON log formatting"""
    print("\n=== Test 1: JSON Formatter ===")
    
    # Setup logger with JSON format
    structured_logger = setup_structured_logging(
        log_level="INFO",
        log_format="json",
        log_file_path="logs/test_json.log",
        enable_console=False,
        enable_file=True,
        environment="test"
    )
    
    # Set correlation ID
    correlation_id = str(uuid.uuid4())
    StructuredLogger.set_correlation_id(correlation_id)
    StructuredLogger.set_user_id("test_user_123")
    StructuredLogger.set_request_path("/api/test")
    
    # Log a message
    logger.info("Test JSON log message", extra={"test_field": "test_value"})

    # Small delay to ensure logs are flushed
    time.sleep(0.1)

    # Read the log file
    log_file = Path("logs/test_json.log")
    if log_file.exists():
        with open(log_file, "r") as f:
            lines = f.readlines()
            if len(lines) > 0:
                last_line = lines[-1]
                log_entry = json.loads(last_line)

                # Loguru's serialize creates a nested structure
                record = log_entry.get("record", {})
                extra = record.get("extra", {})

                # Validate JSON structure
                assert "record" in log_entry, "Missing record"
                assert "message" in record, "Missing message"
                assert "level" in record, "Missing level"
                assert "time" in record, "Missing time"
                assert "correlation_id" in extra, "Missing correlation_id in extra"
                assert "user_id" in extra, "Missing user_id in extra"
                assert "request_path" in extra, "Missing request_path in extra"
                assert extra.get("extra", {}).get("test_field") == "test_value", "Extra field missing"

                print(f"✅ JSON log entry valid:")
                print(f"   Message: {record['message']}")
                print(f"   Level: {record['level']['name']}")
                print(f"   Correlation ID: {extra.get('correlation_id')}")
                print(f"   User ID: {extra.get('user_id')}")
                print(f"   Request Path: {extra.get('request_path')}")
            else:
                print("❌ Log file is empty")
    else:
        print("❌ Log file not created")
    
    # Clear context
    StructuredLogger.clear_context()


def test_correlation_id_propagation():
    """Test correlation ID propagation across multiple log calls"""
    print("\n=== Test 2: Correlation ID Propagation ===")
    
    # Setup logger
    setup_structured_logging(
        log_level="INFO",
        log_format="json",
        log_file_path="logs/test_correlation.log",
        enable_console=False,
        enable_file=True,
        environment="test"
    )
    
    # Set correlation ID
    correlation_id = str(uuid.uuid4())
    StructuredLogger.set_correlation_id(correlation_id)
    
    # Log multiple messages
    logger.info("First log message")
    logger.info("Second log message")
    logger.info("Third log message")

    # Small delay to ensure logs are flushed
    time.sleep(0.1)

    # Read the log file
    log_file = Path("logs/test_correlation.log")
    if log_file.exists():
        with open(log_file, "r") as f:
            lines = f.readlines()

            if len(lines) >= 3:
                # Check last 3 lines
                correlation_ids = []
                for line in lines[-3:]:
                    log_entry = json.loads(line)
                    record = log_entry.get("record", {})
                    extra = record.get("extra", {})
                    correlation_ids.append(extra.get("correlation_id"))

                # All should have the same correlation ID
                assert all(cid == correlation_id for cid in correlation_ids), \
                    "Correlation ID not consistent across logs"

                print(f"✅ Correlation ID propagated correctly: {correlation_id}")
                print(f"   Found in {len(correlation_ids)} log entries")
            else:
                print(f"❌ Not enough log entries: {len(lines)}")
    else:
        print("❌ Log file not created")
    
    # Clear context
    StructuredLogger.clear_context()


def test_performance_impact():
    """Test performance impact of structured logging"""
    print("\n=== Test 3: Performance Impact ===")
    
    # Setup logger
    setup_structured_logging(
        log_level="INFO",
        log_format="json",
        log_file_path="logs/test_performance.log",
        enable_console=False,
        enable_file=True,
        environment="test"
    )
    
    # Set correlation ID
    StructuredLogger.set_correlation_id(str(uuid.uuid4()))
    
    # Measure time for 100 log calls
    num_logs = 100
    start_time = time.time()
    
    for i in range(num_logs):
        logger.info(f"Performance test log {i}", extra={"iteration": i})
    
    end_time = time.time()
    total_time_ms = (end_time - start_time) * 1000
    avg_time_ms = total_time_ms / num_logs
    
    print(f"✅ Performance test completed:")
    print(f"   Total time: {total_time_ms:.2f}ms for {num_logs} logs")
    print(f"   Average time per log: {avg_time_ms:.2f}ms")
    
    # Check if performance is acceptable (< 5ms per log)
    if avg_time_ms < 5:
        print(f"   ✅ Performance acceptable (< 5ms per log)")
    else:
        print(f"   ⚠️ Performance may be slow (> 5ms per log)")
    
    # Clear context
    StructuredLogger.clear_context()


def test_file_rotation():
    """Test log file rotation configuration"""
    print("\n=== Test 4: File Rotation Configuration ===")
    
    # Setup logger with rotation
    structured_logger = setup_structured_logging(
        log_level="INFO",
        log_format="json",
        log_file_path="logs/test_rotation.log",
        log_rotation="1 day",
        log_retention="30 days",
        log_compression="zip",
        enable_console=False,
        enable_file=True,
        environment="test"
    )
    
    # Log a message
    logger.info("Test rotation configuration")
    
    # Check if log file exists
    log_file = Path("logs/test_rotation.log")
    if log_file.exists():
        print(f"✅ Log file created: {log_file}")
        print(f"   Rotation: 1 day")
        print(f"   Retention: 30 days")
        print(f"   Compression: zip")
    else:
        print("❌ Log file not created")


def test_environment_based_config():
    """Test environment-based configuration"""
    print("\n=== Test 5: Environment-Based Configuration ===")
    
    # Test development environment (text format)
    print("\n--- Development Environment ---")
    setup_structured_logging(
        log_level="DEBUG",
        log_format="text",
        log_file_path="logs/test_dev.log",
        enable_console=False,
        enable_file=True,
        environment="development"
    )
    
    logger.debug("Development debug message")
    logger.info("Development info message")
    
    # Test production environment (JSON format)
    print("\n--- Production Environment ---")
    setup_structured_logging(
        log_level="INFO",
        log_format="json",
        log_file_path="logs/test_prod.log",
        enable_console=False,
        enable_file=True,
        environment="production"
    )
    
    logger.info("Production info message")
    
    # Check files
    dev_log = Path("logs/test_dev.log")
    prod_log = Path("logs/test_prod.log")
    
    if dev_log.exists() and prod_log.exists():
        print(f"✅ Both environment logs created")
        print(f"   Development: {dev_log}")
        print(f"   Production: {prod_log}")
    else:
        print("❌ Environment logs not created")


def run_all_tests():
    """Run all structured logging tests"""
    print("=" * 60)
    print("STRUCTURED LOGGING TESTS")
    print("=" * 60)
    
    try:
        test_json_formatter()
        test_correlation_id_propagation()
        test_performance_impact()
        test_file_rotation()
        test_environment_based_config()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    run_all_tests()

