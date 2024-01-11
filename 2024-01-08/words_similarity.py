from gensim.models import Word2Vec

model = Word2Vec.load("w2v_model")
rs = model.wv.most_similar("low".lower())
for r in rs:
    print(r)