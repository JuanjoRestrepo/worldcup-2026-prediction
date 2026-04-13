"""Test API endpoints that don't require database connection."""

import sys

import requests

BASE_URL = "http://localhost:8000"


def test_config_endpoint():
    """Test /config endpoint - should work without DB."""
    try:
        response = requests.get(f"{BASE_URL}/config", timeout=5)
        print(f"✅ /config endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Response keys: {list(data.keys())}")
            return True
        return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to API at {BASE_URL}")
        print(
            "   Make sure to run: uv run python -m uvicorn src.api.main:app --port 8000"
        )
        return False


def test_docs_endpoint():
    """Test /docs endpoint - Swagger UI."""
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        print(f"✅ /docs endpoint: {response.status_code}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def test_openapi_schema():
    """Test /openapi.json endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/openapi.json", timeout=5)
        print(f"✅ /openapi.json schema: {response.status_code}")
        if response.status_code == 200:
            schema = response.json()
            endpoints = list(schema.get("paths", {}).keys())
            print(f"   Available endpoints: {endpoints}")
            return True
        return False
    except requests.exceptions.ConnectionError:
        return False


if __name__ == "__main__":
    print("\n🔍 Testing API endpoints (no DB required):\n")

    results = {
        "config": test_config_endpoint(),
        "docs": test_docs_endpoint(),
        "openapi": test_openapi_schema(),
    }

    print("\n📊 Results:")
    print(f"   /config:     {'✅ PASS' if results['config'] else '❌ FAIL'}")
    print(f"   /docs:       {'✅ PASS' if results['docs'] else '❌ FAIL'}")
    print(f"   /openapi.json: {'✅ PASS' if results['openapi'] else '❌ FAIL'}")

    passed = sum(results.values())
    print(f"\n   {passed}/3 endpoints working ✅\n")

    if not any(results.values()):
        print("❌ API is not responding. Start it with:")
        print("   uv run python -m uvicorn src.api.main:app --port 8000 --reload\n")
        sys.exit(1)
