import os
import dashscope
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain_community.llms.tongyi import Tongyi
from langchain.agents import AgentType
from langchain.chains import ConversationChain

DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
if not DASHSCOPE_API_KEY:
    raise ValueError('Please Input Dash Scope Env')

# 加载模型
dashscope.api_key = DASHSCOPE_API_KEY
llm = Tongyi(model='qwen-turbo', dashscope_api_key=DASHSCOPE_API_KEY)

conversationChain = ConversationChain(llm=llm, verbose=True)

result = conversationChain.predict(input="Hi, there！")
print(result)

result = conversationChain.predict(input="I'm doing well! Just having a conversation aith an AI.")
print(result)