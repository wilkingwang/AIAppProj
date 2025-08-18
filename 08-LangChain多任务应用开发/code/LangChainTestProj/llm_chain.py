import os
import dashscope
from langchain.prompts import PromptTemplate
from langchain_community.llms.tongyi import Tongyi

DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
if not DASHSCOPE_API_KEY:
    raise ValueError('Please Input Dash Scope Env')

dashscope.api_key = DASHSCOPE_API_KEY
llm = Tongyi(model='qwen-turbo', dashscope_api_key=DASHSCOPE_API_KEY)

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}"
)

chain = prompt | llm

# result = chain.invoke({"product": "colorful socks"})
# print(result)

result = chain.invoke({"product": "广告设计"})
print(result)