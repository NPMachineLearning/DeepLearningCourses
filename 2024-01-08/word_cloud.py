# load stopwords
from collections import Counter

import jieba
import numpy as np
from matplotlib import pyplot as plt
from wordcloud import WordCloud
import cv2

with open("./stops.txt", "r", encoding="utf-8") as f:
    stop = set(f.read().split("\n"))

with open("./wc.txt", "r", encoding="utf-8") as f:
    txt = " ".join(f.read().split("\n"))
txt = txt.replace(" ", "").replace("/", "").replace("\"", "")

jieba.set_dictionary("dict.txt")
terms = [t for t in jieba.cut(txt, cut_all=True) if t not in stop]
# print(terms)

mask = cv2.imdecode(np.fromfile("heart.png", dtype=np.uint8), cv2.IMREAD_UNCHANGED)

word_cloud = WordCloud(font_path="simsun.ttc", background_color="white", mask=mask)
image = word_cloud.generate_from_frequencies(frequencies=Counter(terms))
plt.imshow(image, interpolation="bilinear")
plt.axis(False)
plt.show()