import requests

response = requests.post(
    'http://localhost:8080/api/chat',
    json={"prompt": "你好，请你介绍一下自己"}
)

print(response.json())