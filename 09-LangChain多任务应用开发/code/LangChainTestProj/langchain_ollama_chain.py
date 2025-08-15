from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个文本技术文档撰写者"),
    ("user", "{input}")
])

llm = Ollama(model='deepseek-r1:8b')

chin = prompt | llm

chin.invoke({
    "input": "langsmith如何帮助进行测试"
})