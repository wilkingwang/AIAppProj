import json
import os
import base64
import dashscope
from dashscope.api_entities.dashscope_response import Role

api_key = os.environ.get('DASHSCOPE_API_KEY')
dashscope.api_key = api_key

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

base64_image = encode_image("./pdf_table.jpg")

def llm_call(messages):
    response = dashscope.MultiModalConversation.call(
        model='qwen-vl-plus',
        messages=messages
    )

    return response

messages = [
    {
        "role": "system",
         "content": [
            {
                 "text": "You are a helpful assistant."
            }
        ]
    },
    {
        'role': 'user',
        'content': [
            {
                'image': f"data:image/jpeg;base64,{base64_image}"
            },
            {
                'text': '这是一个表格图片，帮我提取里面的内容，输出JSON格式'
            }
        ]
    }
]

response = llm_call(messages)
print(response)