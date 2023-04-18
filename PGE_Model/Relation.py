import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os,sys,glob
import io
from PIL import Image
# 提供多组关系的点状分布图

def Relation_btw_CGPA_and_Score(df):
    plt.scatter(df['Score'],df['CGPA'])
    plt.title('平均绩点与考研成绩的关系')
    plt.xlabel('考研分数')
    plt.ylabel('平均绩点')
    buffer = io.BytesIO()
    plt.savefig(buffer,dpi=400,format='PNG',transparent = True)
    img = Image.open(io.BytesIO(buffer.getvalue()))
    plt.close()
    buffer.close()
    return img

def Relation_btw_Math_and_Score(df):
    plt.subplots_adjust(top=0.91)
    plt.scatter(df['Score'],df['MTH'],color = (81/255,170/255,236/255),marker="x")
    plt.title('数学成绩与考研成绩的关系')
    plt.xlabel('考研分数')
    plt.ylabel('数学分数')
    buffer = io.BytesIO()
    plt.savefig(buffer,dpi=400,format='PNG',transparent = True)
    img = Image.open(io.BytesIO(buffer.getvalue()))
    plt.close()
    buffer.close()
    return img

def Relation_btw_Math_and_Pass(df):
    plt.scatter(df['Pass'],df['MTH'])
    plt.title('数学成绩与成功上岸的关系')
    plt.xlabel('上岸')
    plt.ylabel('数学分数')
    buffer = io.BytesIO()
    plt.savefig(buffer,dpi=400,format='PNG',transparent = True)
    img = Image.open(io.BytesIO(buffer.getvalue()))
    plt.close()
    buffer.close()
    return img

def Relation_btw_CET4_and_ENGscore(df):
    plt.scatter(df['ENG'],df['CET4'])
    plt.title('四级成绩与考研英语成绩的关系')
    plt.xlabel('英语分数')
    plt.ylabel('四级分数')
    buffer = io.BytesIO()
    plt.savefig(buffer,dpi=400,format='PNG',transparent = True)
    img = Image.open(io.BytesIO(buffer.getvalue()))
    plt.close()
    buffer.close()
    return img

def Relation_btw_CET6_and_ENGscore(df):
    df[df['CET6']>0].plot(kind='scatter',x='ENG',y='CET6',color='red')
    plt.title('六级成绩与考研英语成绩的关系')
    plt.xlabel('英语分数')
    plt.ylabel('六级分数')
    buffer = io.BytesIO()
    plt.savefig(buffer,dpi=400,format='PNG',transparent = True)
    img = Image.open(io.BytesIO(buffer.getvalue()))
    plt.close()
    buffer.close()
    return img

def Relation_btw_Score_and_UR(df):
    plt.scatter(df['Score'],df['University Rating'])
    plt.title('学校等级与考研分数的关系')
    plt.xlabel('考研分数')
    plt.ylabel('学校等级')
    buffer = io.BytesIO()
    plt.savefig(buffer,dpi=400,format='PNG',transparent = True)
    img = Image.open(io.BytesIO(buffer.getvalue()))
    plt.close()
    buffer.close()
    return img

def Relation_btw_SOP_and_CGPA(df):
    plt.scatter(df['CGPA'],df['SOP'])
    plt.title('深造意愿与本科绩点的关系')
    plt.xlabel('本科绩点')
    plt.ylabel('深造意愿')
    buffer = io.BytesIO()
    plt.savefig(buffer,dpi=400,format='PNG',transparent = True)
    img = Image.open(io.BytesIO(buffer.getvalue()))
    plt.close()
    buffer.close()
    return img

def Relation_btw_SOP_and_CGPA(df):
    plt.scatter(df['CGPA'],df['SOP'])
    plt.title('深造意愿与本科绩点的关系')
    plt.xlabel('本科绩点')
    plt.ylabel('深造意愿')
    buffer = io.BytesIO()
    plt.savefig(buffer,dpi=400,format='PNG',transparent = True)
    img = Image.open(io.BytesIO(buffer.getvalue()))
    plt.close()
    buffer.close()
    return img

def Relation_btw_SOP_and_GPA(df):
    plt.subplots_adjust(top=0.91)
    plt.scatter(df['CGPA'],df['SOP'],color = (81/255,170/255,236/255),marker="x")
    plt.xlabel('平均绩点')
    plt.ylabel('自身意愿')
    plt.title('平均绩点与自身意愿的关系')
    buffer = io.BytesIO()
    plt.savefig(buffer,dpi=400,format='PNG',transparent = True)
    img = Image.open(io.BytesIO(buffer.getvalue()))
    plt.close()
    buffer.close()
    #img.show()
    return img

def Relation_btw_SUC_and_Res(df):
    plt.subplots_adjust(top=0.91)
    #3plt.scatter(df['Success'],df['Research'],color = 'black',marker="x")
    #print("没有研习经历:",len(df[df.Research == 0]))
    #print("有研习经历:",len(df[df.Research == 1]))
    y = np.array([len(df[df.Research == 0]),len(df[df.Research == 1])])
    x = np.arange(2)
    plt.bar(x,y,edgecolor = (81/255,170/255,236/255),color = 'none')
    plt.title("研习经历与保送的关系")
    plt.xlabel("保送")
    plt.ylabel("人数")
    plt.xticks(x,('没有研习经历','有研习经历'))

    buffer = io.BytesIO()
    plt.savefig(buffer,dpi=400,format='PNG',transparent = True)

    img = Image.open(io.BytesIO(buffer.getvalue()))
    plt.close()
    buffer.close()
    #img.show()
    return img

def Relation_btw_UR_and_CGPA(df):
    plt.scatter(df['University Rating'],df['CGPA'])
    plt.title('本科绩点与学校等级的关系')
    plt.xlabel('学校等级')
    plt.ylabel('本科绩点')
    buffer = io.BytesIO()
    plt.savefig(buffer,dpi=400,format='PNG',transparent = True)
    img = Image.open(io.BytesIO(buffer.getvalue()))
    plt.close()
    buffer.close()
    return img

def Relation_btw_GRE_and_CGPA(df):
    plt.scatter(df['GRE Score'],df['CGPA'])
    plt.title('GRE分数与本科绩点的关系')
    plt.xlabel('GRE分数')
    plt.ylabel('本科绩点')
    buffer = io.BytesIO()
    plt.savefig(buffer,dpi=400,format='PNG',transparent = True)
    img = Image.open(io.BytesIO(buffer.getvalue()))
    plt.close()
    buffer.close()
    return img

def Relation_btw_UR_and_Chance_of_Admit(df):
    s = df[df['Chance of Admit '] >= 0.75]['University Rating'].value_counts().head(5)
    plt.title('学校等级与成功进修的关系')
    s.plot(kind='bar',edgecolor = (81/255,170/255,236/255),color = 'none')
    plt.xlabel('学校等级')
    plt.ylabel('进修')
    buffer = io.BytesIO()
    plt.savefig(buffer,dpi=400,format='PNG',transparent = True)
    img = Image.open(io.BytesIO(buffer.getvalue()))
    plt.close()
    buffer.close()
    return img

def Relation_btw_GRE_and_SOP(df):
    plt.scatter(df['GRE Score'],df['SOP'])
    plt.xlabel('GRE成绩')
    plt.ylabel('自身意愿')
    plt.title('自身意愿与GRE成绩的关系')
    buffer = io.BytesIO()
    plt.savefig(buffer,dpi=400,format='PNG',transparent = True)
    img = Image.open(io.BytesIO(buffer.getvalue()))
    plt.close()
    buffer.close()
    return img

def Relation_btw_GRE_and_TOEFL(df):
    df[df['CGPA']>=8.5].plot(kind='scatter',x='GRE Score',y='TOEFL Score',color='red')
    plt.xlabel('GRE分数')
    plt.ylabel('TOEFL分数')
    plt.title('CGPA>=8.5时GRE与TOEFL的关系')
    plt.grid(True)
    buffer = io.BytesIO()
    plt.savefig(buffer,dpi=400,format='PNG',transparent = True)
    img = Image.open(io.BytesIO(buffer.getvalue()))
    plt.close()
    buffer.close()
    return img

def Relation_btw_SUC_and_UR(df):
    plt.subplots_adjust(top=0.91)
    plt.subplots_adjust(right=0.91)
    s = df[df['Success'] == 1]['University Rating'].value_counts().head(5)
    plt.title('学校等级与成功保送的关系')
    s.plot(kind='bar',edgecolor = (81/255,170/255,236/255),color = 'none')
    plt.xlabel('学校等级')
    plt.ylabel('保送')
    buffer = io.BytesIO()
    plt.savefig(buffer,dpi=400,format='PNG',transparent = True)
    img = Image.open(io.BytesIO(buffer.getvalue()))
    plt.close()
    buffer.close()
    return img

def Relation_btw_SUC_and_CGPA(df):
    plt.scatter(df['Success'],df['CGPA'])
    plt.title('本科均分与保送的关系')
    plt.xlabel('是否保送')
    plt.ylabel('本科均分')
    buffer = io.BytesIO()
    plt.savefig(buffer,dpi=400,format='PNG',transparent = True)
    img = Image.open(io.BytesIO(buffer.getvalue()))
    plt.close()
    buffer.close()
    return img

def Relation_btw_SUC_and_CET4(df):
    plt.scatter(df['Success'],df['CET4'])
    plt.title('CET4与保送的关系')
    plt.xlabel('是否保送')
    plt.ylabel('CET4')
    buffer = io.BytesIO()
    plt.savefig(buffer,dpi=400,format='PNG',transparent = True)
    img = Image.open(io.BytesIO(buffer.getvalue()))
    plt.close()
    buffer.close()
    return img

