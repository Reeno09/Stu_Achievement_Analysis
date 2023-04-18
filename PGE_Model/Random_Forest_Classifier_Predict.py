import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os,sys,glob
import io
from PIL import Image

#上岸预测
def predict_PGE(df,cet4,cet6,ur,SOP,MTH,ENG,Score,KP,Research,Paper,CGPA,DV2CL,DV2SL):

    test_data = {
                 'University Rating':[ur],
                 'SOP':[SOP],
                 'MTH':[MTH],
                 'ENG':[ENG],
                 'Score':[Score],
                 'Keep_PGE':[KP],
                 'Research':[Research],
                 'Paper':[Paper],
                 'CET4':[cet4],
                 'CET6':[cet6],
                 'CGPA':[CGPA],
                 'DV2CL':[DV2CL],
                 'DV2SL':[DV2SL],

                 }
    dft = pd.DataFrame(test_data)
    #df[['CET4','CET6','University Rating','SOP','MTH','ENG','Score','Keep_PGE','Research','Paper','CGPA','DV2CL','DV2SL']] = scaleX.transform(df[['CET4','CET6','University Rating','SOP','MTH','ENG','Score','Keep_PGE','Research','Paper','CGPA','DV2CL','DV2SL']])
    #print('预测结果：',lr.predict(df))
    res = str(Random_Forest_Classifier(df,'PGE',dft))
    return res

#留学预测
def predict_SA(df,GRE,TOEFL,UR,SOP,LOP,Research,CGPA):
    test_data = {'GRE Score':[GRE],
                 'TOEFL Score':[TOEFL],
                 'University Rating':[UR],
                 'SOP':[SOP],
                 'LOR':[LOP],
                 'CGPA':[CGPA],
                 'Research':[Research]

                 }
    dft = pd.DataFrame(test_data)
    res = str(Random_Forest_Classifier(df,'SA',dft))
    return res

#保送预测
def predict_PR(df,CET4,CET6,UR,SOP,Research,CGPA,Paper):
    test_data = {
                 'University Rating':[UR],
                 'SOP':[SOP],
                 'Research':[Research],
                 'Paper':[Paper],
                 'CET4':[CET4],
                 'CET6':[CET6],
                 'CGPA':[CGPA],
                 }
    dft = pd.DataFrame(test_data)
    res = str(Random_Forest_Classifier(df,'PR',dft))
    return res

#随机森林用来判断深造概率

def Random_Forest_Classifier(df,type,dft=None):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    if(type == 'PGE' or type == 'dPGE'or type == 'dPGE1'):
        y = df['Pass'].values
        x = df.drop(['Pass'],axis=1)
    elif(type == 'SA' or type == 'dSA' or type == 'dSA1'):
        #SerialNO = df['Serial No.'].values
        #df.drop(['Serial No.'],axis=1)
        df = df.rename(columns={'Chance of Admit ':'Chance of Admit'})
        y = df['Chance of Admit'].values
        x = df.drop(['Chance of Admit'],axis=1)
    elif(type == 'PR' or type == 'dPR' or type == 'dPR1'):
        y = df['Success'].values
        x = df.drop(['Success'],axis=1)

    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.9,random_state=42)#分割数据集

    from sklearn.preprocessing import MinMaxScaler
    scaleX = MinMaxScaler(feature_range=[0,1])#设定放缩范围
    x_train[x_train.columns] = scaleX.fit_transform(x_train[x_train.columns])#对数据进行放缩
    x_test[x_test.columns] = scaleX.fit_transform(x_test[x_test.columns])
    if(type == 'PGE'):dft[['University Rating','SOP','MTH','ENG','Score','Keep_PGE','Research','Paper','CET4','CET6','CGPA','DV2CL','DV2SL']] = scaleX.transform(dft[['University Rating','SOP','MTH','ENG','Score','Keep_PGE','Research','Paper','CET4','CET6','CGPA','DV2CL','DV2SL']])
    if(type == 'SA'):dft[['GRE Score','TOEFL Score','University Rating','SOP','LOR','CGPA','Research']] = scaleX.transform(dft[['GRE Score','TOEFL Score','University Rating','SOP','LOR','CGPA','Research']])
    if(type == 'PR'):dft[['University Rating','SOP','Research','Paper','CET4','CET6','CGPA']] = scaleX.transform(dft[['University Rating','SOP','Research','Paper','CET4','CET6','CGPA']])

    if(type == 'SA' or type == 'dSA' or type =='dSA1'):
        # 如果chance >0.8, chance of admit 就是1，否则就是0
        y_train_01 = [1 if each > 0.8 else 0 for each in y_train]
        y_test_01 = [1 if each > 0.8 else 0 for each in y_test]

        y_train_01 = np.array(y_train_01)
        y_test_01 = np.array(y_test_01)

    from sklearn.ensemble import RandomForestClassifier#随机森林分类

    rfc = RandomForestClassifier(n_estimators=75,random_state=1)#决策树的个数，越多性能就会越差
    if(type == 'PGE' or type == 'dPGE' or type == 'dPGE1'):
        rfc.fit(x_train,y_train)
        print('rfPGE精度评估: ',rfc.score(x_test,y_test))#精度评估
        print('Real value of y_test_01[1]: '+str(y_test[1]) + ' -> predict value: ' + str(rfc.predict(x_test.iloc[[1],:])))
        print('Real value of y_test_01[2]: '+str(y_test[2]) + ' -> predict value: ' + str(rfc.predict(x_test.iloc[[2],:])))
    elif(type == 'SA' or type == 'dSA' or type =='dSA1'):
        rfc.fit(x_train,y_train_01)
        print('rfSA精度评估: ',rfc.score(x_test,y_test_01))
        print('Real value of y_test_01[1]: '+str(y_test_01[1]) + ' -> predict value: ' + str(rfc.predict(x_test.iloc[[1],:])))
        print('Real value of y_test_01[2]: '+str(y_test_01[2]) + ' -> predict value: ' + str(rfc.predict(x_test.iloc[[2],:])))
    elif(type == 'PR' or type == 'dPR' or type == 'dPR1'):
        rfc.fit(x_train,y_train)
        print('rfPR精度评估: ',rfc.score(x_test,y_test))#精度评估
        print('Real value of y_test_01[1]: '+str(y_test[1]) + ' -> predict value: ' + str(rfc.predict(x_test.iloc[[1],:])))
        print('Real value of y_test_01[2]: '+str(y_test[2]) + ' -> predict value: ' + str(rfc.predict(x_test.iloc[[2],:])))


    from sklearn.metrics import confusion_matrix#混淆矩阵，表明多个类别是否有混淆（是否预测错），为热力图作准备
    if(type == 'PGE' or type == 'dPGE' or type == 'dPGE1'):
        cm_rfc = confusion_matrix(y_test,rfc.predict(x_test))#测试数据集混淆矩阵
    elif(type == 'SA' or type == 'dSA' or type =='dSA1'):
        cm_rfc = confusion_matrix(y_test_01,rfc.predict(x_test))
    elif(type == 'PR' or type == 'dPR' or type == 'dPR1'):
        cm_rfc = confusion_matrix(y_test,rfc.predict(x_test))
    if(type == 'dPGE' or type == 'dSA' or type == 'dPR'):
        f,ax = plt.subplots(figsize=(5,5))
        sns.heatmap(cm_rfc,annot=True,linewidths=0.5,linecolor='red',fmt='.0f',ax=ax)
        plt.title('测试数据集检测')
        plt.xlabel('预测值')
        plt.ylabel('真实值')
        buffer = io.BytesIO()
        plt.savefig(buffer,dpi=400,format='PNG',transparent = True)
        img = Image.open(io.BytesIO(buffer.getvalue()))
        plt.close()
        buffer.close()
        return img

    if(type == 'PGE'or type == 'dPGE'or type == 'dPGE1'):
        cm_rfc_train = confusion_matrix(y_train,rfc.predict(x_train))#训练数据集混淆矩阵
    elif(type == 'SA' or type == 'dSA' or type =='dSA1'):
        cm_rfc_train = confusion_matrix(y_train_01,rfc.predict(x_train))
    elif(type == 'PR' or type == 'dPR' or type == 'dPR1'):
        cm_rfc_train = confusion_matrix(y_train,rfc.predict(x_train))

    if(type == 'dPGE1' or type == 'dSA1' or type == 'dPR1'):
        f,ax = plt.subplots(figsize=(5,5))
        sns.heatmap(cm_rfc_train,annot=True,linewidths=0.5,linecolor='blue',fmt='.0f',ax=ax)
        plt.title('训练数据集检测')
        plt.xlabel('预测值')
        plt.ylabel('真实值')
        buffer = io.BytesIO()
        plt.savefig(buffer,dpi=400,format='PNG',transparent = True)
        img = Image.open(io.BytesIO(buffer.getvalue()))
        plt.close()
        buffer.close()
        return img

    from sklearn.metrics import recall_score,precision_score,f1_score#对二分类问题常用的评估指标是精度(precision)、召回率(recall)、F1值(F1-score)
    if(type == 'PGE'):
        print('precision_score is : ',precision_score(y_test,rfc.predict(x_test)))
        print('recall_score is : ',recall_score(y_test,rfc.predict(x_test)))
        print('f1_score is : ',f1_score(y_test,rfc.predict(x_test)))
        #print(rfc.predict(dft))
        return rfc.predict(dft)
    elif(type == 'SA'):
        from sklearn.metrics import recall_score,precision_score,f1_score
        print('precision_score is : ',precision_score(y_test_01,rfc.predict(x_test)))
        print('recall_score is : ',recall_score(y_test_01,rfc.predict(x_test)))
        print('f1_score is : ',f1_score(y_test_01,rfc.predict(x_test)))
        return rfc.predict(dft)
    elif(type == 'PR'):
        print('precision_score is : ',precision_score(y_test,rfc.predict(x_test)))
        print('recall_score is : ',recall_score(y_test,rfc.predict(x_test)))
        print('f1_score is : ',f1_score(y_test,rfc.predict(x_test)))
        #print(rfc.predict(dft))
        return rfc.predict(dft)