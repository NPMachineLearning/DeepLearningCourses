import re
import jieba
from gensim.models import Word2Vec
import time

TEXT_CLEANING_RE = "[0-9a-zA-Z,:.?!\"\+\-*/_='()\[\]|<>$（）［］，｜、《》！？”%【】“　．…❤️：]"

def preprocess(text):
    text = re.sub(TEXT_CLEANING_RE, " ", str(text).lower()).strip()
    text = text.replace(" ", "")
    return text

jieba.set_dictionary("dict.txt")

with open("stops.txt", "r", encoding="utf-8") as f:
    stopwords = f.read().strip("\n")
stopwords = set(stopwords)

with open("./train.csv/train_tc.csv", "r", encoding="utf-8") as f:
    lines = f.readlines()

vocabularies = []
for line in lines:
    line = preprocess(line)
    terms = [t for t in jieba.cut(line, cut_all=True) if t not in stopwords]
    vocabularies.append(terms)

w2v = Word2Vec(window=7,
               min_count=10,
               workers=8)
w2v.build_vocab(vocabularies)
words = list(w2v.wv.key_to_index.keys())
vocab_size = len(words)
print(f"Total vocabs: {vocab_size}")

print("Vectorizing.....")
t1 = time.time()
w2v.train(vocabularies, total_examples=len(vocabularies), epochs=32)
t2 = time.time()
print(f"Time for vectorization: {t2-t1} seconds")
w2v.save("./w2v_model_chinese")