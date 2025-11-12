#!/usr/bin/env python3
"""
File Upload Diagnostic Script
Tests the file upload functionality to identify issues
"""

import requests
import os
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"
UPLOAD_ENDPOINT = f"{API_BASE_URL}/api/v1/knowledge/upload"

def test_backend_connection():
    """Test if backend is running"""
    print("🔍 Testing backend connection...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running")
            return True
        else:
            print(f"⚠️  Backend returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Backend is not running. Start it with: cd backend && python main.py")
        return False
    except Exception as e:
        print(f"❌ Error connecting to backend: {e}")
        return False

def test_upload_directory():
    """Test if upload directory exists and is writable"""
    print("\n🔍 Testing upload directory...")
    upload_dir = Path("data/uploads")
    
    if not upload_dir.exists():
        print(f"⚠️  Upload directory doesn't exist: {upload_dir}")
        print("   Creating directory...")
        try:
            upload_dir.mkdir(parents=True, exist_ok=True)
            print("✅ Upload directory created")
            return True
        except Exception as e:
            print(f"❌ Failed to create directory: {e}")
            return False
    else:
        print(f"✅ Upload directory exists: {upload_dir}")
        
    # Test write permissions
    test_file = upload_dir / "test_write.txt"
    try:
        test_file.write_text("test")
        test_file.unlink()
        print("✅ Upload directory is writable")
        return True
    except Exception as e:
        print(f"❌ Upload directory is not writable: {e}")
        return False

def test_file_upload():
    """Test actual file upload"""
    print("\n🔍 Testing file upload...")
    
    # Create a test file
    test_content = "This is a test file for upload testing"
    test_filename = "test_upload.txt"
    
    try:
        # Prepare the file for upload
        files = {
            'files': (test_filename, test_content, 'text/plain')
        }
        
        # Optional: Add collection parameter
        data = {
            'collection': 'test'
        }
        
        print(f"   Uploading to: {UPLOAD_ENDPOINT}")
        response = requests.post(
            UPLOAD_ENDPOINT,
            files=files,
            data=data,
            timeout=30
        )
        
        print(f"   Response status: {response.status_code}")
        print(f"   Response body: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"✅ File uploaded successfully!")
                print(f"   Files uploaded: {result.get('total', 0)}")
                if result.get('files'):
                    for file_info in result['files']:
                        print(f"   - {file_info.get('filename')} ({file_info.get('file_size')} bytes)")
                return True
            else:
                print(f"⚠️  Upload succeeded but returned success=False")
                return False
        else:
            print(f"❌ Upload failed with status {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Upload request timed out")
        return False
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False

def test_cors():
    """Test CORS configuration"""
    print("\n🔍 Testing CORS configuration...")
    try:
        response = requests.options(
            UPLOAD_ENDPOINT,
            headers={
                'Origin': 'http://localhost:5173',
                'Access-Control-Request-Method': 'POST',
                'Access-Control-Request-Headers': 'Content-Type'
            },
            timeout=5
        )
        
        cors_headers = {
            'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
            'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods'),
            'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers'),
        }
        
        print(f"   CORS Headers:")
        for header, value in cors_headers.items():
            if value:
                print(f"   ✅ {header}: {value}")
            else:
                print(f"   ⚠️  {header}: Not set")
        
        return True
    except Exception as e:
        print(f"❌ CORS test error: {e}")
        return False

def test_endpoint_availability():
    """Test if the upload endpoint is available"""
    print("\n🔍 Testing endpoint availability...")
    try:
        # Try to access the endpoint with GET (should fail but endpoint should exist)
        response = requests.get(UPLOAD_ENDPOINT, timeout=5)
        
        if response.status_code == 405:  # Method Not Allowed
            print("✅ Upload endpoint exists (returns 405 for GET as expected)")
            return True
        elif response.status_code == 404:
            print("❌ Upload endpoint not found (404)")
            print("   The knowledge router may not be loaded")
            return False
        else:
            print(f"⚠️  Unexpected status code: {response.status_code}")
            return True
    except Exception as e:
        print(f"❌ Endpoint test error: {e}")
        return False

def main():
    """Run all diagnostic tests"""
    print("=" * 60)
    print("FILE UPLOAD DIAGNOSTIC TOOL")
    print("=" * 60)
    
    results = {
        "Backend Connection": test_backend_connection(),
        "Upload Directory": test_upload_directory(),
        "Endpoint Availability": test_endpoint_availability(),
        "CORS Configuration": test_cors(),
        "File Upload": test_file_upload(),
    }
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All tests passed! File upload should be working.")
    else:
        print("\n⚠️  Some tests failed. Please address the issues above.")
        print("\nCommon Solutions:")
        print("1. Make sure backend is running: cd backend && python main.py")
        print("2. Check backend logs for errors")
        print("3. Verify CORS settings in backend/main.py")
        print("4. Ensure data/uploads directory has write permissions")
    
    print("=" * 60)

if __name__ == "__main__":
    main()

