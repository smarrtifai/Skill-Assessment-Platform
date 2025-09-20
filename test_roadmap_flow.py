#!/usr/bin/env python3
"""
Test script to simulate the complete assessment to roadmap flow
"""

import requests
import json

BASE_URL = "http://localhost:5000"

def test_complete_flow():
    """Test the complete flow from assessment to roadmap."""
    
    # Create a session
    session = requests.Session()
    
    print("🧪 Testing complete assessment to roadmap flow...")
    
    # Step 1: Start onboarding
    print("\n1. Testing onboarding...")
    response = session.get(f"{BASE_URL}/")
    if response.status_code == 200:
        print("✅ Homepage loaded")
    else:
        print(f"❌ Homepage failed: {response.status_code}")
        return False
    
    # Step 2: Complete onboarding
    print("\n2. Testing onboarding completion...")
    onboarding_data = {
        "step": 0,
        "response": "I'm new to tech"
    }
    response = session.post(f"{BASE_URL}/api/onboarding", json=onboarding_data)
    if response.status_code == 200:
        print("✅ Onboarding step 1 completed")
    else:
        print(f"❌ Onboarding step 1 failed: {response.status_code}")
        return False
    
    # Step 3: Select interests
    print("\n3. Testing interest selection...")
    interests_data = {
        "step": 1,
        "response": "Machine Learning, Python, Data Science"
    }
    response = session.post(f"{BASE_URL}/api/onboarding", json=interests_data)
    if response.status_code == 200:
        print("✅ Interests selected")
    else:
        print(f"❌ Interest selection failed: {response.status_code}")
        return False
    
    # Step 4: Choose assessment
    print("\n4. Testing assessment choice...")
    assessment_choice = {
        "step": 2,
        "response": "Let me take a quick assessment first"
    }
    response = session.post(f"{BASE_URL}/api/onboarding", json=assessment_choice)
    if response.status_code == 200:
        print("✅ Assessment choice made")
    else:
        print(f"❌ Assessment choice failed: {response.status_code}")
        return False
    
    # Step 5: Start interest-based assessment
    print("\n5. Testing interest-based assessment...")
    response = session.post(f"{BASE_URL}/api/assessment/interest-based", json={})
    if response.status_code == 200:
        print("✅ Interest-based assessment started")
    else:
        print(f"❌ Interest-based assessment failed: {response.status_code}")
        return False
    
    # Step 6: Simulate assessment completion
    print("\n6. Testing assessment completion...")
    assessment_data = {
        "answers": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
        "questions": [{"correct": 0} for _ in range(30)],
        "skills": ["Machine Learning", "Python", "Data Science"]
    }
    response = session.post(f"{BASE_URL}/api/assess", json=assessment_data)
    if response.status_code == 200:
        print("✅ Assessment completed")
    else:
        print(f"❌ Assessment completion failed: {response.status_code}")
        return False
    
    # Step 7: Generate roadmap
    print("\n7. Testing roadmap generation...")
    roadmap_data = {
        "tech_field": "AI/ML"
    }
    response = session.post(f"{BASE_URL}/api/roadmap", json=roadmap_data)
    if response.status_code == 200:
        print("✅ Roadmap generated")
    else:
        print(f"❌ Roadmap generation failed: {response.status_code}")
        return False
    
    # Step 8: Check roadmap data
    print("\n8. Testing roadmap data retrieval...")
    response = session.get(f"{BASE_URL}/api/get_roadmap_data")
    if response.status_code == 200:
        roadmap_data = response.json()
        print(f"✅ Roadmap data retrieved: {roadmap_data}")
        if roadmap_data.get('field'):
            print(f"✅ Roadmap field: {roadmap_data['field']}")
            return True
        else:
            print(f"❌ No field in roadmap data: {roadmap_data}")
            return False
    else:
        print(f"❌ Roadmap data retrieval failed: {response.status_code}")
        return False

if __name__ == "__main__":
    print("🚀 Testing complete assessment to roadmap flow...")
    success = test_complete_flow()
    if success:
        print("\n🎉 Complete flow test passed!")
    else:
        print("\n💥 Complete flow test failed!")
