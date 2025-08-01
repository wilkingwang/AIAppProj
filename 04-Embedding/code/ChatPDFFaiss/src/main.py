import os
import openai
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI
from langchain.llms.tongyi import Tongyi
from langchain_community.callbacks.manager import get_openai_callback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Tuple

openai.api_key = os.getenv('DASHSCOPE_API_KEY')
openai.base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'

def extract_text_with_page_number(pdf) -> Tuple[str, List[int]]:
    """
    从PDF中提取文本并记录每行文本对应的页码

    参数：
        pdf：PDF文件对象

    返回：
        text：提取的文本内容
        page_number：每行文本对应的页码列表
    """
    text = ""
    page_numbers = []

    for page_number, page in enumerate(pdf.pages, start=1):
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
            page_numbers.extend([page_number] * len(extracted_text.split("\n")))
        else:
            print(f"No text found on page {page_number}")

    return text, page_numbers

def process_text_with_splitter(text: str, page_numbers: List[int]) -> FAISS:
    """
    处理文本并创建向量存储

    参数：
        text：提取的文本内容
        page_numbers：每行文本对应的页码列表

    返回：
        knowledgeBase：基于FAISS的向量存储对象
    """
    # 创建文本分割器，用于将长文本分割成小块
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    # 分割文本
    chunks = text_splitter.split_text(text)

    print(f"文本被分割成 {len(chunks)} 个块")

    # 创建嵌入模型
    embeddings = HuggingFaceEmbeddings()

    # 从文本块创建知识库
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    print(f"已从文本块创建知识库")

    knowledgeBase.page_info = {chunk: page_numbers[i] for i, chunk in enumerate(chunks)}
    return knowledgeBase


pdf_reader = PdfReader("../data/浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf")

# 提取文本和页码信息
text, page_numbers = extract_text_with_page_number(pdf_reader)

# 处理文本并创建知识库
knowledgeBase = process_text_with_splitter(text, page_numbers)

query = "客户经理被投诉了，投诉一次扣多少分"
if query:
    # 执行相似度搜索，找到与查询相关的文档
    docs = knowledgeBase.similarity_search(query)

    llm = Tongyi(
        model="qwen-plus",
        api_key='sk-72315aebf5b74cf8952db8173f114c3d',
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat"
    )

    # 加载问答链
    chain = load_qa_chain(llm, chain_type="stuff")

    # 准备输入数据
    input_data = {"input_documents": docs, "question": query}

    # 使用回调函数跟踪API调用成本
    with get_openai_callback() as cost:
        # 执行问答链
        response = chain.invoke(input=input_data)
        print(f"查询已处理，成本: {cost}")
        print(response['output_text'])

        print('来源：')

    unique_pages = set()
    for doc in docs:
        text_content = getattr(doc, "page_content", "")
        source_page = knowledgeBase.page_info.get(
            text_content.strip(), "未知"
        )

        if source_page not in unique_pages:
            unique_pages.add(source_page)
            print(f"文本页码: {source_page}")