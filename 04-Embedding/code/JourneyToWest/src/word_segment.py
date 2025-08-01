# -*-coding: utf-8 -*-
# 对txt文件进行中文分词

import jieba
import os
from utils import files_processing

# 源文件所在目录
source_dir = "../data/source"
segment_dir = "../data/segment"

# 字词分割，对整个文件内容进行字词分割
def segment_files_list(files_list, segment_out_dir, stopwords=[]):
    # 确保输出目录存在
    if not os.path.exists(segment_out_dir):
        os.makedirs(segment_out_dir)
        
    for i, file in enumerate(files_list):
        # 修复文件名格式化错误
        segment_out_file = os.path.join(segment_out_dir, "segment_{}.txt".format(i))

        with open(file, 'rb') as f:
            document = f.read()
            document_cut = jieba.cut(document)
            
            sentences_segment = []
            for word in document_cut:
                # 过滤停用词
                if word not in stopwords:
                    sentences_segment.append(word)

            result = ' '.join(sentences_segment)
            # 修复变量名错误：readlines 应该是 result
            result = result.encode('utf-8')

            with open(segment_out_file, 'wb') as out_file:
                out_file.write(result)

file_list = files_processing.get_files_list(source_dir, postfix='txt')
segment_files_list(file_list, segment_dir)