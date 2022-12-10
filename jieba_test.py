import jieba
import zhon.hanzi
import requests

punc = zhon.hanzi.punctuation  # 要去除的中文标点符号
url = 'http://104.208.89.216:2333/down/nUhzKi1BU4qh'
unit_file = requests.get(url)
open('baidu_stopwords', 'wb').write(unit_file.content)
file_stop = r'baidu_stopwords'
stopwords = []
with open(file_stop, 'r', encoding='utf-8-sig') as f:
    lines = f.readlines()  # lines是list类型
    for line in lines:
        lline = line.strip()  # line 是str类型,strip 去掉\n换行符
        stopwords.append(lline)  # 将stop 是列表形式

# 读入文件
with open('to_be_Slime.txt', encoding="utf-8") as fp:
    text = fp.read()

ls = jieba.lcut(text)  # 分词

# 统计词频
counts = {}
for i in ls:
    if len(i) > 1:
        counts[i] = counts.get(i, 0) + 1

for word in stopwords:  # 去掉停用词
    counts.pop(word, 0)

ls1 = sorted(counts.items(), key=lambda x: x[1], reverse=True)  # 词频排序

for i in range(20):
    print(ls1[i])




from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def create_word_cloud(filename):
    text = open("{}.txt".format(filename), encoding='UTF-8').read()
    wordlist = jieba.cut(text, cut_all=True)
    wl = " ".join(wordlist)
    cloud_mask = np.array(Image.open("giegie.png"))
    wc = WordCloud(
        background_color="black",
        mask=cloud_mask,
        max_words=2000,
        font_path='msyh.ttc',
        height=876,
        width=711,
        max_font_size=100,
        random_state=100,
    )
    myword = wc.generate(wl)
    plt.imshow(myword)
    plt.axis("off")
    plt.show()
    wc.to_file('giegie_book.png')


if __name__ == '__main__':
    create_word_cloud('to_be_Slime')
