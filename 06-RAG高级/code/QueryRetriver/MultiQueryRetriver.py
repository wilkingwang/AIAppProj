import os
from langchain.retrievers import MultiQueryRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi

# 获取环境变量中的 DASHSCOPE_API_KEY
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

llm = Tongyi(model_name="qwen-turbo", dashscope_api_key=DASHSCOPE_API_KEY)

# 创建Embedding模型
embedding = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DASHSCOPE_API_KEY
)

# 加载向量数据库添加allow_dangerous_deserialization参数允许反序列化
vectorstore = FAISS.load_local("./vector_db", embedding, allow_dangerous_deserialization=True)

# 创建MultiQeuryRetriver
retriver = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)

# 执行查询
query = "客户经理的考核标准是什么？"
results = retriver.invoke(query)

print(f'找到 {len(results)} 个相关文档')
for i, doc in enumerate(results):
    print(f'\n文档{i}')
    print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)