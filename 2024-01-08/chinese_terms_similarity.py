from gensim.models import Word2Vec

model = Word2Vec.load("./w2v_model_chinese")
while True:
    voc = input("Input term or quit to exit: ")
    if voc == "quit": break
    try:
        rs = model.wv.most_similar(voc)
        for r in rs:
            print(r)
    except:
        print(f"No similar words with {voc}")