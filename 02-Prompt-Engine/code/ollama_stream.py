import requests

base_url = 'http://localhost:11434/api/generate'

def ollama_call(prompt, model='deepseek-r1:1.5b', stream = False):
    data = {
        'model': model,
        'prompt': prompt,
        'stream': stream
    }

    if stream:
        with requests.post(base_url, json=data, stream=True) as response:
            if response.status_code == 200:
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        try:
                            import json

                            obj = json.loads(line)
                            print(obj.get('response', ''), end='', flush=True)
                        except Exception as e:
                            print(f'Parse Stream Response failed: {str(e)}')
            else:
                print(response)
                raise Exception(f'API request failed: {response.text}')
    else:
        response = requests.post(url=base_url, json=data)
        if response.status_code == 200:
            print(response.json()['response'])
        else:
            raise Exception(f'API request failed: {response.text}')
        
print("Stream response:")
ollama_call("帮我写一个二分查找法", stream=True)