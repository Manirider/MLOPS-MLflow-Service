import requests
import json
import time

def test_optimization():
    url = "http://localhost:8000/predict"
    data = {"image": [0.0] * 784}

    print("Sending first request (Cold Cache)...")
    start = time.time()
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        print(
            f"Response 1: Status {response.status_code}, Time: {time.time() - start:.4f}s")
    except Exception as e:
        print(f"Request 1 failed: {e}")
        return

    print("\nSending second request (Warm Cache)...")
    start = time.time()
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        print(
            f"Response 2: Status {response.status_code}, Time: {time.time() - start:.4f}s")
    except Exception as e:
        print(f"Request 2 failed: {e}")

if __name__ == "__main__":
    test_optimization()
