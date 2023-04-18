import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os,sys,glob
import numpy as np

# 供多组柱状图表示单项数据的最值与平均值

def Draw_UR_W2B(df):
    y = np.array([df['University Rating'].min(),df['University Rating'].mean(),df['University Rating'].max()])
    x = np.arange(3)
    plt.bar(x,y)
    plt.title('University Rating')
    plt.xlabel('Level')
    plt.ylabel('Score')
    plt.xticks(x,('Worst','Average','Best'))
    plt.show()

def Draw_MTH_W2B(df):
    y = np.array([df['MTH'].min(),df['MTH'].mean(),df['MTH'].max()])
    x = np.arange(3)
    plt.bar(x,y)
    plt.title('Math')
    plt.xlabel('Level')
    plt.ylabel('Score')
    plt.xticks(x,('Worst','Average','Best'))
    plt.show()



