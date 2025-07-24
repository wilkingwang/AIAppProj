import dashscope
from dashscope.api_entities.dashscope_response import Role

# 使用指令来指示模型想要实现的内容
instruction_prompt = """
    将下面的文本翻译成英语：
    文本：你好中国
    """

# 给出具体执行指令和任务
specific_prompt = """
从以下文本中提取地名。
期望的输出格式：
地点：<以逗号分隔公司名称的列表>

输入：Although these developments are encouraging to researchers, much is still a mystery. “We often have a black box between the brain and the effect we see in the periphery,” says Henrique Veiga-Fernandes, a neuroimmunologist at the Champalimaud Centre for the Unknown in Lisbon. “If we want to use it in the therapeutic context, we need to understand the mechanism.
"""

# 具体直接给出有效信息
accurate_prompt = """
使用2-3句话向高中生解释提示工程的概念。
"""

# 提示词可以更具体、并侧重于细节，这些细节可以引导模型产生良好的响应
todo_prompt = """
以下是向客户推荐电影的代理。代理应该从全球热门电影中推荐电影给客户。它不应该询问用户的偏好并避免询问个人信息。如果代理没有电影可以推荐，它应该回复“对不起，今天找不到电影可以推荐。”。

客户：请根据我的兴趣推荐一部电影。

"""

def base_prompt():
    messages = [
        {
            "role": "user",
            "content": todo_prompt
        }
    ]

    response = dashscope.Generation.call(
        model='qwen-turbo',
        messages=messages,
        result_format='message'
    )

    print(response.output.choices[0].message.content)

def main():
    base_prompt()

if __name__ == '__main__':
    main()