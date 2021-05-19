# -- coding:utf-8 --

# import sys
# import importlib
# import codecs
# importlib.reload(sys)
# sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# mpl.use("pgf")
# pgf_config = {
#     "font.family":'serif',
#     "font.size": 7.5,
#     "pgf.rcfonts": False,
#     "text.usetex": True,
#     "pgf.preamble": [
#         r"\usepackage{unicode-math}",
#         r"\setmainfont{Times New Roman}",
#         r"\usepackage{xeCJK}",
#         r"\setCJKmainfont{SimSun}",
#     ],
# }
# plt.rcParams.update(pgf_config)


def split_name(f):
    assert type(f) == str
    tmp = f.split('.jpg.')[0]
    return [int(tmp.split('_')[i]) for i in range(3)]


datadir = r"./data/UTKFace/"
images = os.listdir(datadir)
data = []

for i in images:
    info = split_name(i)
    data.append(info)
data = pd.DataFrame(data)
data.columns = ['Age', 'Gender', 'Race']
data['Age'] = data['Age'].clip(0, 100)

females = np.array(data[data.Gender == 1]['Age'])
males = np.array(data[data.Gender == 0]['Age'])
white = np.array(data[data.Race == 0]['Age'])
black = np.array(data[data.Race == 1]['Age'])
asian = np.array(data[data.Race == 2]['Age'])
indian = np.array(data[data.Race == 3]['Age'])
others = np.array(data[data.Race == 4]['Age'])



# Songti = mpl.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
plt.rcParams['font.sans-serif'] = ['STSong']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
x1 = [males, females]
x2 = [white, black, asian, indian, others]
n_bins = 10
ax0.hist(x1, n_bins, histtype='bar', label=['男性', '女性'])
ax0.set_title('按性别的数据分布', fontsize=14)
ax0.set_xticks([5,15,25,35,45,55,65,75,85,95])
ax0.set_xlabel('年龄/岁')
ax0.set_ylabel('样本量/个')
ax0.legend()
ax1.hist(x2, n_bins, histtype='bar', label=['白人', '黑人', '亚裔', '印地', '其他'])
ax1.set_title('按人种的数据分布', fontsize=14)
ax1.set_xticks([5,15,25,35,45,55,65,75,85,95])
ax1.set_xlabel('年龄/岁')
ax1.legend()
plt.show()