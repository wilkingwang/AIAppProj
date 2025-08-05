import os
import pickle
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms.tongyi import Tongyi
from typing import List, Tuple

DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
if not DASHSCOPE_API_KEY:
    raise ValueError('Please Input Dash Scope Env')

def extract_text_with_page_number(pdf) -> Tuple[str, List[Tuple[str, int]]]:
    """
    从PDF中提取文本并记录每个字符对应的页码
    参数：
        pdf：PDF文件帝乡
    返回：
        text：提取的文本内容
        char_page_mapping：每个字符对应的页码列表
    """
    text = ""
    char_page_mapping = []

    for page_number, page in enumerate(pdf.pages, start=1):
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
            # 为当前页面的每个字符记录页码
            char_page_mapping.extend([page_number] * len(extracted_text))
        else:
            print(f"No text found on page {page_number}")

    return text, char_page_mapping

def process_text_with_splitter(text: str, char_page_mapping: List[int], save_path: str = None) -> FAISS:
    """
    处理文本并创建向量存储
    参数：
        text：提取的文本内容
        char_page_mapping：每个字符对应的页码列表
        save_path：可选，保存向量数据库路径
    返回：
        knowledgeBase：给予FAISS的向量存储对象
    """
    # 创建文本分割器，用于将长文本分割成小块
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ",", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    # 分割文本
    chunks = text_splitter.split_text(text)
    print(f"文本被分割成 {len(chunks)} 个块。")

    # 创建嵌入模型
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=DASHSCOPE_API_KEY
    )

    # 从文本块中创建知识库
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    print("已从文本块创建知识库。")

    # 为每个文本找到对应的页码信息
    page_info = {}
    current_pos = 0

    for chunk in chunks:
        chunk_start = current_pos
        chunk_end = current_pos + len(chunk)

        # 找到这个文本块中字符对应的页码
        chunk_pages = char_page_mapping[chunk_start:chunk_end]

        # 取页码的众数（出现最多的页码），作为该块的页码
        if chunk_pages:
            # 统计每个页码出现的次数
            page_counts = {}
            for page in chunk_pages:
                page_counts[page] = page_counts.get(page, 0) + 1

            # 找到出现次数最多的页码
            most_common_page = max(page_counts, key=page_counts.get)
            page_info[chunk] = most_common_page
        else:
            page_info[chunk] = 1

        current_pos = chunk_end

    knowledgeBase.page_info = page_info
    print(f"页码映射完成，共 {len(page_info)} 个文本块")

    # 如果提供了保存路径，则保存向量数据库也页码信息
    if save_path:
        os.makedirs(save_path, exist_ok=True)

        # 保存FAISS向量数据库
        knowledgeBase.save_local(save_path)
        print(f"向量数据库已保存到: {save_path}")

        # 保存页码信息到同一目录
        with open(os.path.join(save_path, "page_info.pkl"), "wb") as f:
            pickle.dump(page_info, f)
            print(f"页码信息已保存到：{os.path.join(save_path, "page_info.pkl")}")

    return knowledgeBase

def load_knowledge_base(load_path: str, embeddings = None) -> FAISS:
    """
    从磁盘加载向量数据库和页码信息
    参数：
        load_path：向量数据库的保存路径
        embeddinds：可选，嵌入模型。如果为None，将创建一个新的DashScopeEmbeddings实例

    返回：
        knowledgeBase：加载的FAISS向量数据库对象
    """
    if embeddings is None:
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v1",
            dashscope_api_key=DASHSCOPE_API_KEY
        )

    # 加载FAISS向量数据库，添加allow_dangerous_deserialization=True参数允许反序列化
    knowledgeBase = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
    print(f"向量数据库已从 {load_path} 加载.")

    # 加载页码信息
    page_info_path = os.path.join(load_path, "page_info.pkl")
    if os.path.exists(page_info_path):
        with open(page_info_path, "rb") as f:
            page_info = pickle.load(f)

        knowledgeBase.page_info = page_info
        print(f"页码信息已加载。")
    else:
        print("警告，未找到页码信息文件。")
    
    return knowledgeBase

pdf_reader = PdfReader('../data/浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf')

text, char_page_mapping = extract_text_with_page_number(pdf_reader)

print(f'提取的文本长度: {len(text)} 个字符.')

save_dir = './data/vector_db'
knowledgeBase = process_text_with_splitter(text, char_page_mapping, save_path=save_dir)

# embeddings = DashScopeEmbeddings(
#     model="text-embedding-v1",
#     dashscope_api_key=DASHSCOPE_API_KEY
# )

# loaded_knowledge_base = load_knowledge_base(save_dir, embeddings)

# docs = load_knowledge_base.similarity_search("客户经理每年评聘申报时间是怎样的？")

llm = Tongyi(model="deepseek-v3", dashscope_api_key=DASHSCOPE_API_KEY)

print(DASHSCOPE_API_KEY)
input_text = "用50个字左右阐述，生命的意义在于"
llm.invoke(input_text)

# query = "客户经理被投诉了，投诉一次扣多少分"
# if query:
#     docs = knowledgeBase.similarity_search(query, k=10)

#     chain = load_qa_chain(llm, chain_type="stuff")

#     input_data = {"input_documents": docs, "question":query}

#     response = chain.invoke(input=input_data)
#     print(response['output_text'])

#     print("来源：")
#     unique_pages = set()
#     for doc in docs:
#         text_content = getattr(doc, "page_content", "")
#         source_page = knowledgeBase.page_info.get(
#             text_content.strip(), "未知"
#         )

#         if source_page not in unique_pages:
#             unique_pages.add(source_page)
#             print(f'文本块页码：{source_page}')