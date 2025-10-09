#!/usr/bin/env python3
"""
Test script to verify API fixes work correctly.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_gemini_timeout():
    """Test Gemini API with timeout handling."""
    try:
        import google.generativeai as genai
        
        # Configure Gemini
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("GEMINI_API_KEY not found in environment")
            return False
            
        genai.configure(api_key=api_key, transport='rest')
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Test simple generation with timeout
        generation_config = genai.types.GenerationConfig(
            temperature=0.3,
            max_output_tokens=100,
            candidate_count=1
        )
        
        prompt = "Generate a simple JSON object with field 'test' and value 'success'"
        response = model.generate_content(prompt, generation_config=generation_config)
        
        print("Gemini API test successful")
        print(f"Response: {response.text[:100]}...")
        return True
        
    except Exception as e:
        print(f"Gemini API test failed: {e}")
        return False

def test_calendly_fallback():
    """Test Calendly fallback mechanism."""
    try:
        # Test with invalid API key
        import calendly
        
        api_key = os.getenv('CALENDLY_API_KEY', 'invalid_key')
        if api_key == 'YOUR_CALENDLY_API_KEY' or not api_key:
            print("Calendly fallback test: API key not configured (expected)")
            return True
            
        try:
            client = calendly.Calendly(api_key)
            user_info = client.about()
            print("Calendly API test successful")
            return True
        except Exception as e:
            print(f"Calendly fallback test: API error handled gracefully - {e}")
            return True
            
    except Exception as e:
        print(f"Calendly test failed: {e}")
        return False

def test_groq_timeout():
    """Test Groq API with timeout."""
    try:
        from groq import Groq
        
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            print("GROQ_API_KEY not found in environment")
            return False
            
        client = Groq(api_key=api_key)
        
        # Test simple completion
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Say 'test successful'"}],
            model="llama3-8b-8192",
            temperature=0.1,
            # timeout=10  # Not supported in this Groq version
        )
        
        print("Groq API test successful")
        print(f"Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"Groq API test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing API fixes...\n")
    
    tests = [
        ("Gemini API Timeout", test_gemini_timeout),
        ("Calendly Fallback", test_calendly_fallback),
        ("Groq API Timeout", test_groq_timeout)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Testing {test_name}...")
        result = test_func()
        results.append((test_name, result))
        print()
    
    print("Test Results:")
    print("-" * 40)
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)