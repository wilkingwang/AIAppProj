import os
import pickle
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
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
        chunk_pages = char_page_mapping[chunk_start, chunk_end]

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

