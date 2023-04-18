import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os,sys,glob
import io
from PIL import Image
# 统计各分数段出现频次柱状图

def Draw_Score(df):
    df['Score'].plot(kind='hist',bins=200,figsize=(6,6))
    plt.title('分数统计')
    plt.xlabel('分数')
    plt.ylabel('出现频率')
    buffer = io.BytesIO()
    plt.savefig(buffer,dpi=400,format='PNG',transparent = True)
    img = Image.open(io.BytesIO(buffer.getvalue()))
    plt.close()
    buffer.close()
    return img

def Draw_SA(df):
    df['GRE Score'].plot(kind='hist',bins=200,figsize=(6,6))
    plt.title('GRE分数统计')
    plt.xlabel('GRE分数')
    plt.ylabel('出现频次')
    buffer = io.BytesIO()
    plt.savefig(buffer,dpi=400,format='PNG',transparent = True)
    img = Image.open(io.BytesIO(buffer.getvalue()))
    plt.close()
    buffer.close()
    return img

def Draw_PR(df):
    df['SOP'].plot(kind='hist',bins=200,figsize=(6,6))
    plt.title('立志保送学生的深造意愿')
    plt.xlabel('深造意愿')
    plt.ylabel('出现频次')
    buffer = io.BytesIO()
    plt.savefig(buffer,dpi=400,format='PNG',transparent = True)
    img = Image.open(io.BytesIO(buffer.getvalue()))
    plt.close()
    buffer.close()
    return img