# -- coding:utf-8 --


import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def split_name(f):
    assert type(f) == str
    tmp = f.split('.jpg.')[0]
    return [int(tmp.split('_')[i]) for i in range(3)]


# datadir = r"./analysis/"
# models = os.listdir(datadir)
# print(models)

listfile = []
for root, dirs, files in os.walk("./analysis/", topdown=False):
    for name in files:
        listfile.append(os.path.join(root, name))
    # for name in dirs:
    #     print(os.path.join(root, name))

# print(listfile)
listfile = np.array(listfile).reshape(4,int(len(listfile)/4))
# print(listfile[2])
# listfile[0] is SN-D
# listfile[1] is SN-DE
# listfile[2] is SN-DEH
# listfile[3] is SN-DH
# 0['./analysis/SN-DEH\\run-.-tag-D_img_loss_G.csv'
# 1 './analysis/SN-DEH\\run-.-tag-D_img_loss_input.csv'
# 2 './analysis/SN-DEH\\run-.-tag-D_z_loss_prior.csv'
# 3 './analysis/SN-DEH\\run-.-tag-D_z_loss_z.csv'
# 4 './analysis/SN-DEH\\run-.-tag-EG_loss.csv'
# 5 './analysis/SN-DEH\\run-.-tag-E_z_loss.csv'
# 6 './analysis/SN-DEH\\run-.-tag-G_img_loss.csv'
# 7 './analysis/SN-DEH\\run-.-tag-MAE.csv'
# 8 './analysis/SN-DEH\\run-.-tag-PSNR_1.csv'
# 9 './analysis/SN-DEH\\run-.-tag-SSIM.csv']

import seaborn as sns
# sns.set()
for i in range(10):
    scalar = listfile[0,i]
    scalar = scalar.split('-tag-')[1].split('.csv')[0]
    if scalar=='PSNR_1':
        scalar='PSNR'
    print(scalar)

    fig = plt.figure(figsize=(8,6))

    for file in listfile[:,i]:  
        f = open(file, encoding='UTF-8')
        data = pd.read_csv(f,sep=',',encoding='UTF-8')
        name = file.split('\\')[0].split('/')[2]
        value = data['Value']
        if i < 7:
            value = np.array(value)
            value = value[::10]
        epoch = np.arange(len(value))
        # print(data.shape)    #(75,)
        plt.plot(epoch, value, label=name)
    plt.legend()
    plt.title(scalar)
    plt.savefig('./snmodel-{}.png'.format(scalar))
    plt.close()