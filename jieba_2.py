import jieba
import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openpyxl
from cnsenti import Sentiment

# 需要分析的数据量，最大样本数为102190，如果期望三分钟内完成分析建议设置在10000以内
MAXLINE = 102100

import os
import requests

# 文件的保存路径
file_path = "./stopwords"
# 如果文件不存在，就下载文件
if not os.path.exists(file_path):
    # 文件的下载地址
    url = "http://104.208.89.216:2333/down/nUhzKi1BU4qh"
    # 从服务器下载文件
    r = requests.get(url)
    # 保存文件
    with open(file_path, "wb") as f:
        f.write(r.content)

file_stop = r'stopwords'
stopwords = []
with open(file_stop, 'r', encoding='utf-8-sig') as f:
    lines = f.readlines()  # lines是list类型
    for line in lines:
        lline = line.strip()  # line 是str类型,strip 去掉\n换行符
        stopwords.append(lline)  # 将stop 是列表形式

# 读入文件
data = pd.read_excel('super_cxk.xlsx', header=0)
txt = data['微博内容'][:MAXLINE]

# 用于统计词频
counts = {}

# 用于生成词云
wordlist = ''
flag_num = 1
err_num = 0

# 情感分数统计
senti_sum = [0 for i in range(21)]

for text in txt:
    # print(text)
    flag_num += 1
    if flag_num % 10 == 0:
        print(flag_num)
    try:
        ls = jieba.lcut(text)  # 分词

        for i in ls:  # 统计词频
            if len(i) > 1:
                unit = i
                if unit == '徐坤' or unit == 'cxk':
                    unit = '蔡徐坤'
                if unit == '徐坤元' or unit == '宇宙':
                    unit = '元宇宙'
                if unit == 'prada':
                    unit = 'PRADA'
                counts[unit] = counts.get(unit, 0) + 1
                wordlist += ' ' + unit

        senti = Sentiment()
        result = senti.sentiment_count(text)
        ans = (result['pos'] - result['neg']) // result['sentences']
        if ans > 10:
            ans = 10
        elif ans < -10:
            ans = -10
        senti_sum[ans + 10] += 1
    except:
        print('数据非法！第', flag_num, '条')
        err_num += 1
        continue

# 去掉停用词
for word in stopwords:
    counts.pop(word, 0)

print('共读取到数据：', flag_num, '条')
if err_num != 0:
    print('其中非法数据共有：', err_num, '条')

ls1 = sorted(counts.items(), key=lambda x: x[1], reverse=True)  # 词频排序

# 输出词频前20
for i in ls1[:20]:
    print(i)

from wordcloud import WordCloud
from PIL import Image

# 输出关键词云图
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
myword = wc.generate(wordlist)
plt.imshow(myword)
plt.axis("off")
plt.show()
wc.to_file('giegie_cloud.png')

# 话题关键字统计
# 中文乱码解决方法
plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
# 取出列表中每一行的第一个元素，作为新的切片
slice1 = [row[0] for row in ls1[:20]]
# 取出列表中每一行的第二个元素，作为新的切片
slice2 = [row[1] for row in ls1[:20]]
# 使用 pyplot 模块中的 bar() 方法绘制条形图
# 传入字典的键和值来作为图表的数据
plt.bar(slice1, slice2)
plt.xticks(rotation='vertical')
plt.subplots_adjust(bottom=0.2)
# 添加 x 轴和 y 轴的标签
plt.xlabel('话题关键字')
plt.ylabel('出现次数')
# 显示图表
plt.show()

# 帖子情感统计
# 传入字典的键和值来作为图表的数据
plt.bar([i for i in range(-10, 11)], senti_sum)
# plt.xticks(rotation='vertical')
plt.subplots_adjust(bottom=0.2)
# 添加 x 轴和 y 轴的标签
plt.xlabel('情感偏向')
plt.ylabel('相关帖子数量')
# 显示图表
plt.show()

# 创建一个字典，用于统计每个名字出现的次数
name_counts = {}
# 遍历名字列表，统计每个名字出现的次数
for name in data['用户名称'][:MAXLINE]:
    if name in name_counts:
        name_counts[name] += 1
    else:
        name_counts[name] = 1
# 把统计结果按照出现次数排序
sorted_name_counts = sorted(name_counts.items(), key=lambda item: item[1], reverse=True)

# 取出列表中每一行的第一个元素，作为新的切片
man1 = [row[0] for row in sorted_name_counts[:20]]
# 取出列表中每一行的第二个元素，作为新的切片
man2 = [row[1] for row in sorted_name_counts[:20]]
# 传入字典的键和值来作为图表的数据
plt.figure(figsize=(6, 7))
plt.bar(man1, man2)
plt.xticks(fontsize=7, rotation='vertical')
plt.subplots_adjust(bottom=0.2)
# 添加 x 轴和 y 轴的标签
plt.xlabel('用户名称')
plt.ylabel('发帖数量')
# 显示图表
plt.show()

# 创建一个字典，其中包含用户等级和它出现的次数
user = {'普通用户': 0, '黄v': 0, '金v': 0, '蓝v': 0}

for lv in data['微博等级'][:MAXLINE]:
    if lv in user:
        user[lv] += 1

# 创建饼状图
plt.pie(user.values(), labels=user.keys(), autopct='%1.1f%%')

# 显示图表
plt.show()


