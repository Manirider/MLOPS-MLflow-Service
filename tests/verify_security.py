import requests
import time
import sys

API_URL = "http://localhost:8000"
API_KEY = "mlops-secret-key-123"

def test_api_key():
    print("Testing API Key Protection...")

    response = requests.post(f"{API_URL}/predict", json={"image": [0.0]*784})
    if response.status_code == 403:
        print("✅ Protected endpoint rejected request without key (403 Access Forbidden)")
    else:
        print(f"❌ Protected endpoint failed to reject request without key. Status: {response.status_code}")
        return False

    headers = {"X-API-Key": API_KEY}
    response = requests.post(f"{API_URL}/predict", json={"image": [0.0]*784}, headers=headers)

    if response.status_code in [200, 404]:
        print(f"✅ Protected endpoint accepted request with key (Status: {response.status_code})")
    else:
        print(f"❌ Protected endpoint rejected request with valid key. Status: {response.status_code}")
        print(response.text)
        return False

    return True

def test_rate_limiting():
    print("\nTesting Rate Limiting (100 req/min)...")
    headers = {"X-API-Key": API_KEY}

    start_time = time.time()
    for i in range(110):
        response = requests.get(f"{API_URL}/experiments", headers=headers)
        if response.status_code == 429:
            print(f"✅ Rate limit triggered after {i} requests (429 Too Many Requests)")
            return True
        if i % 20 == 0:
            print(f"Sent {i} requests...")

    print("❌ Failed to trigger rate limit.")
    return False

if __name__ == "__main__":
    if test_api_key() and test_rate_limiting():
        print("\n✅ All Security Tests Passed!")
        sys.exit(0)
    else:
        print("\n❌ Security Tests Failed!")
        sys.exit(1)
