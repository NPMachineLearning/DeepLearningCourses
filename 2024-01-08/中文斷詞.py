# 結巴斷詞
# https://github.com/fxsjy/jieba
# 中文繁體結巴詞 https://github.com/APCLab/jieba-tw
# 中文 stopwords https://github.com/GoatWang/ithome_ironman/tree/master/day16_NLP_Chinese
import jieba

jieba.set_dictionary("./dict.txt")

s = [t for t in jieba.cut("下雨天留客我不留")]
print(s)

s = [t for t in jieba.cut("下雨天留客我不留", cut_all=True)]
print(s)
