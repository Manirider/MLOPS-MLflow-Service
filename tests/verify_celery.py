import requests
import time
import sys
import uuid
API_URL = "http://localhost:8000"
API_KEY = "mlops-secret-key-123"
def test_distributed_training():
    print("Testing Distributed Training (Celery)...")
    headers = {"X-API-Key": API_KEY}
    payload = {
        "learning_rate": 0.001,
        "epochs": 1,
        "batch_size": 32,
        "hidden_size": 64,
        "dropout": 0.1,
        "run_name": f"test_celery_run_{uuid.uuid4().hex[:8]}"
    }
    print(f"Submitting training job: {payload}")
    response = requests.post(f"{API_URL}/train", json=payload, headers=headers)
    if response.status_code not in [200, 202]:
        print(f"❌ Failed to submit job. Status: {response.status_code}")
        print(response.text)
        return False
    data = response.json()
    job_id = data.get("job_id")
    print(f"Job submitted. Response: {data}")
    if not job_id:
        print("⚠️ Job ID not found in response! Checking fallback...")
        job_id = data.get("id") or data.get("run_id")
    print(f"Tracking Job ID: {job_id}")
    for _ in range(30):
        time.sleep(2)
        status_res = requests.get(f"{API_URL}/train/status/{job_id}", headers=headers)
        if status_res.status_code == 200:
            status_data = status_res.json()
            status = status_data.get("status")
            print(f"Job Status: {status}")
            if status == "completed":
                print(f"✅ Training completed! Run ID: {status_data.get('run_id')}")
                return True
            elif status == "failed":
                print(f"❌ Training failed: {status_data.get('error')}")
                return False
        else:
            print(f"⚠️ Failed to get status: {status_res.status_code}")
    print("❌ Timeout waiting for training to complete")
    return False
if __name__ == "__main__":
    if test_distributed_training():
        sys.exit(0)
    else:
        sys.exit(1)
