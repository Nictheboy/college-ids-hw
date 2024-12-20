import jieba
import wordcloud
#from scipy.misc import imread
from matplotlib.pyplot import imread

mask = imread("China.jpg")

f = open("2020年政府工作报告.txt", "r", encoding="utf-8")

exclude = {'我们','和','的','今年','万亿元'}  
t = f.read()
f.close()
ls = jieba.lcut(t)
     
txt = " ".join(ls)
font='C:/Windows/Fonts/simfang.ttf'
w = wordcloud.WordCloud( \
    width = 1000, height = 700,\
    background_color = "white",\
    font_path=font,\
    mask=mask,\
    stopwords=exclude
    )
w.generate(txt)
w.to_file("政府工作报告词云.png")
