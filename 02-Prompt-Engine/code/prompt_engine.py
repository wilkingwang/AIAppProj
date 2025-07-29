import os
import dashscope

dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

def get_completion(prompt, model='deepseek-v3'):
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    response = dashscope.Generation.call(
        model=model,
        messages=messages,
        result_format='message',
        temperature=0
    )

    return response.output.choices[0].message.content

instruction = """
你的任务是识别用户对手机流量套餐产品的选择条件。
每种流量套餐产品包含三个属性：名称，月费价格，月流量。
根据用户输入，识别用户在上述三种属性上的需求是什么。
"""

input_text = """
办个100G的套餐。
"""

prompt = f"""
# 目标
{instruction}

# 用户输入
{input_text}
"""

print("========= Prompt ============")
print(prompt)
print("========= Prompt ============")

response = get_completion(prompt)
print(response)

# JSON格式
output_format = """
以 JSON格式输出
"""

prompt = f"""
# 目标
{instruction}

# 输出格式
{output_format}

# 用户输入
{input_text}
"""
response = get_completion(prompt)
print(response)

# CoT示例
instruction = """
给定一段用户与手机流量套餐客服的对话，
你的任务是判断客服的回答是否符合下面的规范：

- 必须礼貌
- 必须用官方口吻，不能使用网络用语
- 介绍套餐时，必须准确提及产品名称、月费价格和月流量总量。上述信息缺失一项或多项，或信息与事实不符，都算信息不准确
- 不可以是话题终结者

已知产品包括：

经济套餐：月费50元，月流量10G
畅想套餐：月费180元，月流量100G
无限套餐：月费300元，月流量1000G
校园套餐：月费150元，月流量200G，限在校学生办理
"""

output_format = """
如果符合规范，输出：Y
如果不符合规范，输出：N
"""

context = """
用户：你们有什么流量大的套餐
客户：亲，我们现在正在推广无限套餐，每月300元就可以享受1000G流量，您感兴趣吗？
"""

cot = ""
#cot = "请一步一步分析对话"
""

prompt = f"""
# 目标
{instruction}
{cot}

# 输出格式
{output_format}

# 对话上下文
{context}
"""

response = get_completion(prompt)
print(response)

# 使用Prompt调优Prompt
