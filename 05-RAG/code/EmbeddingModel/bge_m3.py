from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel(model_name_or_path='D:/05Models/bge-m3', use_fp16=True)

sentences1 = ['Waht is BGE M3', 'Defination of BM25']
sentences2 = ['BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.',
              'BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document']

embedding1 = model.encode(sentences1, batch_size=12, max_length=8192)['dense_vecs']
embedding2 = model.encode(sentences2)['dense_vecs']

similarity = embedding1 @ embedding2.T

print(similarity)