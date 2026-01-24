import json
import hashlib
import redis

def populate_cache():

    image = [0.0] * 784
    input_str = json.dumps(image)
    cache_key = hashlib.md5(f"predict:{input_str}".encode()).hexdigest()

    r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

    mock_response = {
        "prediction": 5,
        "confidence": 0.99,
        "probabilities": [0.0]*5 + [0.99] + [0.0]*4,
        "model_name": "MockModel",
        "model_version": "1",
        "model_stage": "Production"
    }

    r.set(cache_key, json.dumps(mock_response))
    print(f"Set cache for key: {cache_key}")

if __name__ == "__main__":
    populate_cache()
