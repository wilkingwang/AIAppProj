import json
import os
import dashscope
from dashscope.api_entities.dashscope_response import Role

dashscope.api_key = "sk-f85d4acdec0b42b2a46fc87ee877a19c"

def main():
    review = '这款音效特别好 给你意想不到的音质。'

    messages = [
        {
            "role": "system",
            "content": "你是一名舆情分析师，帮我判断产品口碑的正负向，回复请用一个词语：正向 或者 负向"
        },
        {
            "role": "user",
            "content": review
        }
    ]

    response = dashscope.Generation.call(
        model='qwen-turbo',
        messages=messages,
        result_format='message'
    )

    contnet = response.output.choices[0].message.content
    print(response.output.choices[0].message.content)

    messages.append({
        "role": "assistant",
        "content": contnet
    })
    messages.append({
        "role": "user",
        "content": "这个音箱效果不是很好"
    })
    response = dashscope.Generation.call(
        model='qwen-turbo',
        messages=messages,
        result_format='message'
    )

    contnet = response.output.choices[0].message.content
    print(response.output.choices[0].message.content)

if __name__ == "__main__":
    main()