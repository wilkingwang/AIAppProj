import os
import json
import dashscope

# 获取环境变量中的 DASHSCOPE_API_KEY
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

# 基于Prompt生成文本
def get_completion(prompt, model="qwen-turbo-latest"):
    messages =[
        {
            "role": "user",
            "content": prompt
        }
    ]

    response = dashscope.Generation.call(
        model=model,
        messages=messages,
        result_format="message",
        temperature=0
    )

    return response.output.choices[0].message.content


# Query改写
class QueryRewriter:
    def __init__(self, model="qwen-turbo-latest"):
        self.model = model

    def rewriter_conext_dependent_query(self, current_query, conversation_history):
        """
        上下文依赖改写
        """
        instruction = """
你是一个智能的查询优化助手，请分析用户的当前问题以及前序对话历史，判断当前问题是否依赖上下文，
如果依赖，请将当前问题改写成一个独立的、包含所有必要上下文信息的完整问题。
如果不依赖，直接返回原问题。
"""
        prompt = f"""
### 指令 ###
{instruction}

### 对话历史/上下文信息 ###
{conversation_history}

### 原始问题 ###
{current_query}

### 改写后的问题 ###
"""
        return get_completion(prompt, self.model)
    

def main():
    rewriter = QueryRewriter()

    print(f"===Query改写功能示例（上下文依赖）===\n")
    conversation_history = """
用户: "我想了解一下上海迪士尼乐园的最新项目。"
AI: "上海迪士尼乐园最新推出了'疯狂动物城'主题园区，这里有朱迪警官和尼克狐的互动体验。"
用户: "这个园区有什么游乐设施？"
AI: "'疯狂动物城'园区目前有疯狂动物城警察局、朱迪警官训练营和尼克狐的冰淇淋店等设施。"
"""
    current_query = "还有其他设施吗？"
    result = rewriter.rewriter_conext_dependent_query(current_query, conversation_history)
    print(f"改写结果：f{result}")


if __name__ == "__main__":
    main() 