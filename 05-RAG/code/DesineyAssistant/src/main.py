import os
import re
import numpy as np
import faiss
import fitz
import pytesseract
import torch
from openai import OpenAI
from docx import Document as DocxDocument
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# step0：全局配置和模型加载

# 检查环境变量
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("错误: 请设置 DASHSCOPE_API_KEY 环境变量.")

# 初始化百炼兼容的OpenAI客户端
client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

print("正在加载 CLIP 模型...")
try:
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print("CLIP 模型加载成功。")
except Exception as ex:
    print(f"加载 CLIP 模型失败，请检查网络连接或 Hugging Face Token.错误: {ex}")
    exit()

# 定义全局变量
DOCS_DIR = "D:/01Workspace/w30006808/AIAppProj/05-RAG/code/DesineyAssistant/data/disney_knowledge_base"
IMG_DIR = os.path.join(DOCS_DIR, "images")
TEXT_EMBEDDING_MODEL = "text-embedding-v4"
TEXT_EMBEDDING_DIM = 1024
IMAGE_EMBEDDING_DIM = 512

# step1:文档解析与内容提取
def parse_docx(file_path):
    """
    解析DOCS文件，提取文本和表格（转为Markdown）
    """
    doc = DocxDocument(file_path)
    content_chunks = []

    for element in doc.element.body:
        if element.tag.endswith('p'):
            # 段落处理
            paragraph_text = ""
            for run in element.findall('.//w:t', {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}):
                paragraph_text += run.text if run.text else ""

            if paragraph_text.strip():
                content_chunks.append({"type": "text", "content": paragraph_text.strip()})
        elif element.tag.endswith('tbl'):
            # 处理表格
            md_table = []
            table = [t for t in doc.tables if t._element is element][0]
            
            if table.rows:
                # 添加表头
                header = [cell.text.strip() for cell in table.rows[0].cells]
                md_table.append("| " + " | ".join(header) + " |")
                md_table.append("|" + "---|"*len(header))

                # 添加数据行
                for row in table.rows[1:]:
                    row_data = [cell.text.strip() for cell in row.cells]
                    md_table.append("| " + " | ".join(row_data) + " |")

                table_content = "\n".join(md_table)
                if table_content.strip():
                    content_chunks.append({"type": "table", "content": table_content})

    return content_chunks

def image_to_text(image_path):
    """
    对图片进行OCR和CLIP描述
    """
    try:
        image = Image.open(image_path)

        ocr_text = pytesseract.image_to_string(image, lang='chi_sim+eng').strip()
        return {"ocr": ocr_text}
    except Exception as ex:
        print(f"处理图片失败 {image_path}: {ex}")
        return {"ocr": ""}
    
# step2: Embedding与索引构建
def get_text_embedding(text):
    """
    获取文本的Embedding
    """
    response = client.embeddings.create(
        model=TEXT_EMBEDDING_MODEL,
        input=text,
        dimensions=TEXT_EMBEDDING_DIM
    )

    return response.data[0].embedding

def get_image_embedding(image_path):
    """
    获取图片的Embedding
    """
    image = Image.open(image_path)
    inputs = clip_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)

    return image_features[0].numpy()

def get_clip_text_embedding(text):
    """
    使用CLIP的文本编辑器获取文本的Embedding
    """
    inputs = clip_processor(text=text, return_tensors="pt")
    
    while torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)

    return text_features[0].numpy()

def build_knowledge_base(docs_dir, image_dir):
    """
    构建完整的知识库，包括解析、切片、Embedding和索引
    """
    print("\n-------步骤1 & 2: 正在解析、Embedding并索引知识库-------")
    metadata_store = []
    text_vectors = []
    image_vectors = []

    doc_id_counter = 0

    # 处理word文档
    for filename in os.listdir(docs_dir):
        if filename.startswith('.') or os.path.isdir(os.path.join(docs_dir, filename)):
            continue

        file_path = os.path.join(docs_dir, filename)
        if filename.endswith(".docx"):
            print(f"---正在处理：{filename}")
            chunks = parse_docx(file_path)

            for chunk in chunks:
                metadata = {
                    "id": doc_id_counter,
                    "source": filename,
                    "page": 1
                }

                if chunk["type"] == "text" or chunk["type"] == "teble":
                    text = chunk["content"]

                    if not text.strip():
                        continue

                    metadata["type"] = "text"
                    metadata["content"] = text

                    vector = get_text_embedding(text)
                    text_vectors.append(vector)
                    metadata_store.append(metadata)
                    doc_id_counter += 1

    # 处理images目录中的独立图片文件
    print(f'--- 正在处理独立图片文件---')
    for image_filename in os.listdir(image_dir):
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(image_dir, image_filename)
            print(f"---处理图片：{image_filename}")

            image_text_info = image_to_text(image_path)

            metadata = {
                "id": doc_id_counter,
                "source": f"独立图片：{image_filename}",
                "type": "image",
                "path": image_path,
                "ocr": image_text_info["ocr"],
                "page": 1
            }

            vector = get_image_embedding(image_path)
            image_vectors.append(vector)
            metadata_store.append(metadata)
            doc_id_counter += 1

    # 创建FAISS索引
    # 文本索引
    text_index = faiss.IndexFlatL2(TEXT_EMBEDDING_DIM)
    text_index_map = faiss.IndexIDMap(text_index)
    text_ids = [m["id"] for m in metadata_store if m["type"] == "text"]
    # 只有当有文本向量时才添加到索引
    if text_vectors:
        text_index_map.add_with_ids(np.array(text_vectors).astype('float32'), np.array(text_ids))

    # 图像索引
    image_index = faiss.IndexFlatL2(IMAGE_EMBEDDING_DIM)
    image_index_map = faiss.IndexIDMap(image_index)
    image_ids = [m["id"] for m in metadata_store if m["type"] == "image"]
    # 只有当有图像时才添加到索引
    if image_vectors:
        image_index_map.add_with_ids(np.array(image_vectors).astype('float32'), np.array(image_ids))

    print(f'索引构建完成。共索引 {len(text_vectors)} 个文本片段和 {len(image_vectors)} 张图片')
    return metadata_store, text_index_map, image_index_map

# step3: RAG问答流程
def rag_ask(query, metadata_store, text_index, image_index, k=3):
    """
    执行完整的 RAG 流程：检索 -> 构建Prompt -> 生成答案
    """
    # 步骤1：检索
    print("---步骤1：向量化查询并进行检索...")
    retrieved_context = []

    # 文本检索
    query_text_vec = np.array([get_text_embedding(query)]).astype('float32')
    distances, text_ids = text_index.search(query_text_vec, k)
    for i, doc_id in enumerate(text_ids[0]):
        if doc_id != -1:
            # 通过ID在元数据中查找
            match = next((item for item in metadata_store if item["id"] == doc_id), None)
            if match:
                retrieved_context.append(match)
                print(f"---文本检索命中 (ID: {doc_id}, 距离： {distances[0][i]:.4f})")

    # 图像检索(使用CLIP文本编码器)
    # 简单判断是否需要检索图片
    if any(keyword in query.lower() for keyword in ["海报", "图片", "长什么样", "看看", "万圣节", "聚在一起"]):
        print("  - 检测到图像查询关键词，执行图像检索...")
        query_clip_vec = np.array([get_clip_text_embedding(query)]).astype('float32')
        # 只找最相关的1张图
        distances, image_ids = image_index.search(query_clip_vec, 1) 
        for i, doc_id in enumerate(image_ids[0]):
            if doc_id != -1:
                match = next((item for item in metadata_store if item["id"] == doc_id), None)
                if match:
                    # 将OCR内容也加入上下文
                    context_text = f"找到一张相关图片，图片路径: {match['path']}。图片上的文字是: '{match['ocr']}'"
                    retrieved_context.append({"type": "image_context", "content": context_text, "metadata": match})
                    print(f"---图像检索命中 (ID: {doc_id}, 距离: {distances[0][i]:.4f})")

    # 步骤2：构建Prompt，并生成答案
    print("---步骤2：构建Prompt...")
    context_str = ""
    for i, item in enumerate(retrieved_context):
        content = item.get('content', '')
        source = item.get('metadata', {}).get('source', item.get('source', '未知来源'))
        context_str += f"背景知识 {i+1} (来源： {source}): \n{content}\n\n"

    prompt = f"""你是一个迪士尼客服助手。请根据以下背景知识，用友好和专业的语气回答用户的问题。请只使用背景知识中的信息，不要自行发挥。

[背景知识]
{context_str}

[用户问题]
{query}
"""
    print("---步骤3：调用LLM生成最终答案...")
    try:
        completion = client.chat.completions.create(
            model="qwen-plus", # 使用一个强大的模型进行生成
            messages=[
                {"role": "system", "content": "你是一个迪士尼客服助手。"},
                {"role": "user", "content": prompt}
            ]
        )

        final_answer = completion.choices[0].message.content

        # 答案后处理：如果上下文中包含图片，提示用户
        image_path_found = None
        for item in retrieved_context:
            if item.get("type") == "image_context":
                image_path_found = item.get("metadata", {}).get("path")
                break
        
        if image_path_found:
            final_answer += f"\n\n(同时，我为您找到了相关图片，路径为: {image_path_found})"
    except Exception as e:
        final_answer = f"调用LLM时出错: {e}"

    print("\n--- 最终答案 ---")
    print(final_answer)
    return final_answer

if __name__ == '__main__':
    # 1. 构建知识库 (一次性离线过程)
    metadata_store, text_index, image_index = build_knowledge_base(DOCS_DIR, IMG_DIR)

    # 2. 开始问答
    print("\n=============================================")
    print("迪士尼客服RAG助手已准备就绪，开始模拟提问。")
    print("=============================================")
    
    # 案例1: 文本问答
    rag_ask(
        query="我想了解一下迪士尼门票的退款流程",
        metadata_store=metadata_store,
        text_index=text_index,
        image_index=image_index
    )