import os
import dashscope
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain_community.llms.tongyi import Tongyi
from langchain.agents import AgentType

DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
if not DASHSCOPE_API_KEY:
    raise ValueError('Please Input Dash Scope Env')

# 加载模型
dashscope.api_key = DASHSCOPE_API_KEY
llm = Tongyi(model='qwen-turbo', dashscope_api_key=DASHSCOPE_API_KEY)

# 加载 serpapi工具
tools = load_tools(["serpapi", "llm-math"], llm=llm)

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

result = agent.invoke({"input": "当前西安的温度是多少摄氏度？这个温度的1/4是多少？"})
print(result)