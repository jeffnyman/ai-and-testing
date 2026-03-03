import requests

response = requests.post(
  "http://localhost:11434/api/generate",
  json={
    "model": "qwen2.5:latest",
    "prompt": "hello",
    "stream": False,
  },
)

print(response.status_code)
print(response.json())
