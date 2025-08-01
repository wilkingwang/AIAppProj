from gensim.models import word2vec
import multiprocessing

segment_dir = "../data/segment"

# 切分之后的句子子集
sentences = word2vec.PathLineSentences(segment_dir)

# 设置模型参数，进行训练
model2 = word2vec.Word2Vec(sentences, vector_size=256, window=5, min_count=5, workers=multiprocessing.cpu_count())
model2.save('../model/word2vec.model')

print(model2.wv.most_similar(positive=['曹操', '刘备'], negative=['张飞']))
