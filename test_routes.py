#!/usr/bin/env python3
"""
Simple test script to verify all routes are working correctly.
Run this after starting the Flask application.
"""

import requests
import sys

BASE_URL = "http://localhost:5000"

def test_route(path, expected_status=200):
    """Test a route and return success status."""
    try:
        response = requests.get(f"{BASE_URL}{path}")
        if response.status_code == expected_status:
            print(f"✅ {path} - Status: {response.status_code}")
            return True
        else:
            print(f"❌ {path} - Expected: {expected_status}, Got: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ {path} - Connection failed (is the server running?)")
        return False
    except Exception as e:
        print(f"❌ {path} - Error: {e}")
        return False

def main():
    print("🧪 Testing Skills Assessment Platform Routes")
    print("=" * 50)
    
    routes_to_test = [
        "/",
        "/upload", 
        "/assessment",
        "/results",
        "/roadmap"
    ]
    
    passed = 0
    total = len(routes_to_test)
    
    for route in routes_to_test:
        if test_route(route):
            passed += 1
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} routes passed")
    
    if passed == total:
        print("🎉 All routes are working correctly!")
        return 0
    else:
        print("⚠️  Some routes failed. Check the server logs.")
        return 1

if __name__ == "__main__":
    sys.exit(main())