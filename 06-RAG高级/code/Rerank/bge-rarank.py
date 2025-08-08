import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("D:/04Models/bge-reranker-base")
model = AutoModelForSequenceClassification.from_pretrained("D:/04Models/bge-reranker-base")
model.eval()

pairs = [
    ['what is panda?', 'The giant panda is a bear species endemic to China.'],  # 高相关
    ['what is panda?', 'Pandas are cute.'],                                     # 中等相关
    ['what is panda?', 'The Eiffel Tower is in Paris.']                        # 不相关
]

inputs = tokenizer(pairs, padding=True, truncation=True, return_tensor='pt')
scores = model(**inputs).logits.view(-1).float()
print(scores)
