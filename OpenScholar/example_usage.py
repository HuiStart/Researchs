#!/usr/bin/env python3
"""
OpenScholar API Usage Example

Before running this script, please ensure:
1. vLLM model services are started
2. OpenScholar API service is started
3. Valid Semantic Scholar API key
"""

import requests
import json
import time
'''
    功能与作用: API
    调用示例脚本，演示如何请求单条/批量问题、带标题检索、检查服务状态 
  等。
'''
# API Configuration
API_BASE_URL = "http://localhost:38015"
BATCH_ASK_ENDPOINT = f"{API_BASE_URL}/batch_ask"

def test_single_question():
    """Test single question"""
    print("=== Testing Single Question ===")
    
    question = "How do retrieval-augmented language models perform in knowledge-intensive tasks?"
    
    data = {
        "questions": [question]
    }
    
    print(f"Question: {question}")
    print("Sending request...")
    
    try:
        start_time = time.time()
        response = requests.post(BATCH_ASK_ENDPOINT, json=data, timeout=300)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"Request successful! Time taken: {end_time - start_time:.2f} seconds")
            print(f"Keywords used: {result['results'][0]['keywords']}")
            print(f"Answer: {result['results'][0]['output'][:500]}...")
            print(f"Total cost: {result['results'][0]['total_cost']}")
            
        else:
            print(f"Request failed: {response.status_code}")
            print(f"Error message: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Request exception: {e}")

def test_multiple_questions():
    """Test multiple questions"""
    print("\n=== Testing Multiple Questions ===")
    
    questions = [
        "What are the latest developments in transformer architectures?",
        "How do large language models handle few-shot learning?",
        "What are the current challenges in neural machine translation?"
    ]
    
    data = {
        "questions": questions
    }
    
    print(f"Number of questions: {len(questions)}")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")
    
    print("Sending batch request...")
    
    try:
        start_time = time.time()
        response = requests.post(BATCH_ASK_ENDPOINT, json=data, timeout=600)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"Batch request successful! Total time: {end_time - start_time:.2f} seconds")
            
            for i, res in enumerate(result['results'], 1):
                print(f"\n--- Question {i} Results ---")
                print(f"Keywords: {res['keywords']}")
                print(f"Answer: {res['output'][:200]}...")
                print(f"Cost: {res['total_cost']}")
                
        else:
            print(f"Batch request failed: {response.status_code}")
            print(f"Error message: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Request exception: {e}")

def test_with_titles():
    """Test with paper titles"""
    print("\n=== Testing with Paper Titles ===")
    
    data = {
        "questions": ["How does this paper approach the problem of domain adaptation?"],
        "titles": ["Domain-Adversarial Training of Neural Networks"]
    }
    
    print("Sending request with titles...")
    
    try:
        response = requests.post(BATCH_ASK_ENDPOINT, json=data, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            print("Request with titles successful!")
            print(f"Answer: {result['results'][0]['output'][:300]}...")
        else:
            print(f"Request with titles failed: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"Request exception: {e}")

def check_api_status():
    """Check API service status"""
    print("=== Checking API Service Status ===")
    
    try:
        # Send a simple health check request
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code == 404:  # Flask default 404 indicates service is running
            print("✅ API service is running")
            return True
        else:
            print(f"⚠️  API service status abnormal: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API service, please ensure service is started")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Error checking API status: {e}")
        return False

def main():
    """Main function"""
    print("OpenScholar API Usage Example")
    print("=" * 50)
    
    # Check API status
    if not check_api_status():
        print("\nPlease follow these steps to start services:")
        print("1. Start model services: ./start_models.sh (Linux/Mac) or start_models.bat (Windows)")
        print("2. Start API service: python openscholar_api.py --s2_api_key YOUR_API_KEY --reranker_path /path/to/reranker")
        return
    
    # Run tests
    test_single_question()
    test_multiple_questions()
    test_with_titles()
    
    print("\n" + "=" * 50)
    print("Testing complete!")

if __name__ == "__main__":
    main() 