import json
import os
import dashscope
from dashscope.api_entities.dashscope_response import Role

api_key = os.environ.get('DASHSCOPE_API_KEY')
dashscope.api_key = api_key

def get_current_weather(location, unit="摄氏度"):
    temperature = -1

    if '大连' in location or 'Dalian' in location:
        temperature = 10
    
    if '上海' in location or 'Shanghai' in location:
        temperature = 10

    if '深圳' in location or 'Shenzhen' in location:
        temperature = 10
    
    weather_info = {
        "location": location,
        "temperature": temperature,
        "unit": unit,
        "forecast": ["晴天", "微风"]
    }

    return json.dumps(weather_info)

def llm_call(messages):
    try:
        response = dashscope.Generation.call(
            model='qwen-max',
            messages=messages,
            functions=functions,
            result_format='message'
        )

        return response
    except Exception as ex:
        print(f"LLM API调用失败：{str(ex)}")
        return None

def run_conversation(prompt):
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    response = llm_call(messages)
    if not response or not response.output:
        print("获取LLM响应失败")
        return None
    
    print(response)

    message = response.output.choices[0].message
    messages.append(message)
    print('message=', message)

    if hasattr(message, 'function_call') and message.function_call:
        function_call = message.function_call
        tool_name = function_call['name']

        arguments = json.loads(function_call['arguments'])
        print('arguments=', arguments)

        tool_response = get_current_weather(
            location=arguments.get('location'),
            unit=arguments.get('unit')
        )

        tool_info = {
            "role": "function",
            "name": tool_name,
            "content": tool_response
        }
        print('tool_info=', tool_info)
        messages.append(tool_info)
        print('messages=', messages)

        # step4: 得到第二次响应
        response = llm_call(messages)
        if not response or not response.output:
            print("获取LLM第二次响应失败")
            return None
        
        print('response=', response)
        message = response.output.choices[0].message
        return message
    
functions = [
    {
        'name': 'get_current_weather',
        'description': 'Get the current weather in a given location.',
        'parameters': {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description': 'The city and state, e.g. San Francisco, CA'
                },
                'unit': {
                    'type': 'string',
                    'enum': [
                        'celsius',
                        'fahrenheit'
                    ]
                }
            },
            'required': ['location']
        }
    }
]

def main():
    result = run_conversation('大连的天气怎样')
    if result:
        print('最终结果: ', result)
    else:
        print('对话执行失败')

if __name__ == '__main__':
    main()
