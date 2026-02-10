import requests
import json

BASE_URL = "http://localhost:8000"

def test_predict(text):
    url = f"{BASE_URL}/predict"
    payload = {"text": text}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print(f"Input: '{text}'")
        print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server. Is it running?")
    except Exception as e:
        print(f"Error: {e}")

def test_feedback(text, label):
    url = f"{BASE_URL}/feedback"
    payload = {"text": text, "label": label}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print(f"Feedback for '{text}' as {label}")
        print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("--- Testing Prediction Endpoint ---")
    test_predict("create an image of a red dragon") # IMAGE
    test_predict("what is the capital of Indonesia?") # TEXT
    test_predict("find me a recipe for rendang")      # SEARCH
    test_predict("make a video of a dancing cat")     # VIDEO
    
    print("--- Testing Feedback Endpoint ---")
    test_feedback("draw a cyberpunk city", "IMAGE")
