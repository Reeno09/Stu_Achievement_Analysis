import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os,sys,glob
import io
from PIL import Image
# 绘制热力图
def Draw_HeatMap(df):
    df = df.rename(columns={'Success':'成功保送','KS':'选择深造','WPGE':'选择考研','WWORK':'选择工作','WSA':'选择出国','WPR':'选择保研','WSD':'选择直博','Straight2Doctorate':'直博率','Keep_PGE':'二战率',' Postgraduate_Recommendation':'保研率','Pass':'是否通过考研','SL':'院线','CL':'国家线','Score':'考研总分','Serial No.':'序号','Chance of Admit ': '深造概率','University Rating':'学校等级','SOP':'自身意愿','LOR ':'推荐信','CGPA':'加权平均分','Research':'研习经历','MTH':'数学分数','ENG':'英语分数','Paper':'学术产物','DV2CL':'超国线分差','DV2SL':'超院线分差'})
    fig, ax = plt.subplots(figsize=(10, 10))#fig创建绘画区域，ax为当前绘画区域，fig，ax是简写方式，拆开时ax要调用多次重新指定画图区域，figsize设置图形大小
    sb.heatmap(df.corr(), ax=ax, annot=True, linewidths=0.05, fmt='.2f', cmap='coolwarm',mask=False,vmin=-0.75)#df.corr相关性分析函数，ax指定子图位置，cmap热力图颜色
    buffer = io.BytesIO()
    fig.subplots_adjust(top=0.94)
    fig.subplots_adjust(right=1.03)
    plt.title('相关特征热力图')
    plt.savefig(buffer,dpi=300,format='PNG',transparent = True)
    img = Image.open(io.BytesIO(buffer.getvalue()))
    plt.close()
    buffer.close()
    return img

if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    df = pd.read_csv("/Users/reeno/Documents/Achievement_Analysis/Score/Admission_Predict_Ver2.3.csv", sep =",")
    print('There are ', len(df.columns), 'columns')
    for c in df.columns:
        sys.stdout.write(str(c) + ', ')
    df = df.rename(columns={'KS':'选择深造','WPGE':'选择考研','WWORK':'选择工作','WSA':'选择出国','WPR':'选择保研','WSD':'选择直博','Straight2Doctorate':'直博率','Keep_PGE':'二战率',' Postgraduate_Recommendation':'保研率','Pass':'是否通过考研','SL':'院线','CL':'国家线','Score':'考研总分','Serial No.':'序号','Chance of Admit ': '深造概率','University Rating':'学校等级','SOP':'自身意愿','LOR ':'推荐信','CGPA':'加权平均分','Research':'研习经历','MTH':'数学分数','ENG':'英语分数','Paper':'学术产物','DV2CL':'超国线分差','DV2SL':'超院线分差'})
    df.info()
