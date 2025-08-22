import os
import json
import requests
import dashscope
from http import HTTPStatus

from requests.utils import to_key_val_list

dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

weather_tool = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. 北京",
                },
                "adcode": {
                    "type": "string",
                    "description": "The city code, e.g. 110000（北京）",
                }
            },
            "required": ["location"],
        },
    }
}

def get_weather_from_amap(location: str, adcode: str = None):
    url = "https://restapi.amap.com/v3/weather/weatherInfo"
    params = {
        "key": os.getenv('AMAP_API_KEY'),
        "city": adcode if adcode else location,
        "extensions": "base"
    }
    response = requests.get(url, params=params)
    if response.status_code == HTTPStatus.OK:
        return response.json()
    else:
        return {"error": f"Failed to fetch weather: {response.status_code}"}

def run_weather_query():
    messages = [
        {
            "role": "system",
            "content": "你是一个天气查询助手"
        },
        {
            "role": "user",
            "content": "北京现在的天气怎么样"
        }
    ]

    print('---第一次调用LLM---')
    response = dashscope.Generation.call(
        model='qwen-turbo',
        messages=messages,
        tools=[weather_tool],
        tool_choice='auto',
    )

    if response.status_code == HTTPStatus.OK:
        tool_map = {
            'get_current_weather': get_weather_from_amap
        }

        # 从响应中获取消息
        assistant_messages = response.output.choices[0].message

        # 检查是否需要调用工具
        if hasattr(assistant_messages, 'tool_calls') and assistant_messages.tool_calls:
            print('---需要调用工具---')

            # 转换 assistant 消息为标准字典格式
            assistant_dict = {
                "role": "assistant",
                "content": assistant_messages.content if hasattr(assistant_messages, 'content') else None,
            }

            if hasattr(assistant_messages, 'tool_calls'):
                assistant_dict['tool_calls'] = assistant_messages.tool_calls

                tool_response_messages = []
                for tool_call in assistant_messages.tool_calls:
                    func_name = tool_call['function']['name']
                    func_args = json.loads(tool_call['function']['arguments'])

                    print(f'处理工具调用： {func_name}')
                    if func_name in tool_map:
                        function_to_call = tool_map[func_name]
                        function_response = function_to_call(**func_args)
                        tool_response_messages.append({
                            "role": "tool",
                            "content": json.dumps(function_response),
                            "tool_call_id": tool_call['id']
                        })

                update_messages = messages + [assistant_dict] + tool_response_messages
                print('---第二次调用LLM---')
                response = dashscope.Generation.call(
                    model='qwen-turbo',
                    messages=update_messages,
                    tools=[weather_tool],
                    tool_choice='auto',
                )

                if response.status_code == HTTPStatus.OK:
                    print("最终回复：", response.output.choices[0].message.content)
                else:
                    print(f"调用失败: {response.status_code}, {response.message}")

            else:
                print('---不需要调用工具---')
                print("最终回复：", response.output.choices[0].message.content)
        else:
            print('---不需要调用工具---')
            print("最终回复：", response.output.choices[0].message.content)
    else:
        print(f"调用失败: {response.status_code}, {response.message}")

if __name__ == '__main__':
    run_weather_query()
