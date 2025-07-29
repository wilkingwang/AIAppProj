import requests

base_url = 'http://localhost:11434/api/generate'

def ollama_call(prompt, model='deepseek-r1:1.5b'):
    data = {
        'model': model,
        'prompt': prompt,
        'stream': False
    }

    response = requests.post(base_url, json=data)
    if response.status_code == 200:
        return response.json()['response']
    else:
        raise Exception(f'API request failed: {response.text}')
    
response = ollama_call('你好，请介绍一下自己')
print(response)