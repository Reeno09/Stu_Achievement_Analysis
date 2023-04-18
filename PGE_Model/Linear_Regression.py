import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import os,sys,glob
import io
from PIL import Image


def score(a,b,dimension):
    # a is predict, b is actual. dimension is len(train[0]).
    aa=a.copy(); bb=b.copy()
    if len(aa)!=len(bb):
        print('not same length')
        return np.nan

    cc=aa-bb
    wcpfh=sum(cc**2)

    # RR means R_Square
    RR=1-sum((bb-aa)**2)/sum((bb-np.mean(bb))**2)

    n=len(aa); p=dimension
    Adjust_RR=1-(1-RR)*(n-1)/(n-p-1)
    # Adjust_RR means Adjust_R_Square

    return Adjust_RR

#  线性回归模型用来预测分数
def predict_score(scaleX, lr,cet4,cet6,ur):

    test_data = {'CET4':[cet4],
                 'CET6':[cet6],
                 'University Rating':[ur]
                 }
    df = pd.DataFrame(test_data)
    df[['CET4','CET6']] = scaleX.transform(df[['CET4','CET6']])
    #print('预测结果：',lr.predict(df))
    return lr.predict(df)


def Linear_Regression(df,type,cet4=None,cet6=None,ur=None):
    # 读取数据集
    #df = pd.read_csv('/Users/reeno/Documents/Achievement_Analysis/Score/WPGE_Ver1.32.csv', sep=',')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    if(type == 'PGE' or type =='dPGE'):
        df=df.loc[:,['ENG','CET4','CET6','University Rating']]

        y = df['ENG'].values  # y单独保存考研英语成绩
        x = df.drop(['ENG'],axis=1)  # x保存四六级成绩
    else :
        df = df.rename(columns={'Chance of Admit ':'Chance of Admit'})
        y = df['Chance of Admit'].values
        x = df.drop(['Chance of Admit'],axis=1)

    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.9,random_state=42)

    from sklearn.preprocessing import MinMaxScaler  # 对数据进行放缩
    scaleX = MinMaxScaler(feature_range=[0,1])
    if(type == 'PGE' or type == 'dPGE'):
        x_train[['CET4','CET6']] = scaleX.fit_transform(x_train[['CET4','CET6']])#修改只缩放前两列
        x_test[['CET4','CET6']] = scaleX.fit_transform(x_test[['CET4','CET6']])
    else :
        x_train[x_train.columns] = scaleX.fit_transform(x_train[x_train.columns])
        x_test[x_test.columns] = scaleX.fit_transform(x_test[x_test.columns])

    from sklearn.linear_model import LinearRegression  # 线性回归模型
    lr = LinearRegression()  # 创建回归模型
    lr.fit(x_train,y_train)  # 训练模型
    y_test_predict = lr.predict(x_test)  # x_test的预测结果

    print('Real value of y_test[1]: '+str(y_test[1]) + ' -> predict value: ' + str(lr.predict(x_test.iloc[[1],:])))
    print('Real value of y_test[2]: '+str(y_test[2]) + ' -> predict value: ' + str(lr.predict(x_test.iloc[[2],:])))
    print('Real value of y_test[3]: '+str(y_test[3]) + ' -> predict value: ' + str(lr.predict(x_test.iloc[[3],:])))

    from sklearn.metrics import r2_score
    print('R^2 Score: ',r2_score(y_test,y_test_predict))  # R方表示模型的拟合程度
    y_train_predict = lr.predict(x_train)  # x_train的预测结果
    print('R^2 Score(train data):',r2_score(y_train,y_train_predict))
    #print(lr.score(y_train.reshape(-1,1),y_train_predict))
    if(type == 'PGE' or type == 'dPGE'):print(score(y_test,y_test_predict,3))
    else:print(score(y_test,y_test_predict,9))

    #  绘制测试数据集测试点状图
    if(type == 'dPGE' or type == 'dSA'):
        '''
        plt.plot(range(len(y_test_predict)),sorted(y_test_predict),c="black",label= "Predict")
        plt.scatter(np.arange(0,50),y_test[0:50],color='blue')
        #plt.plot(range(len(y_test)),sorted(y_test),c="red",label = "Data")
        plt.legend()
        '''
        from sklearn.linear_model import SGDRegressor
        sgd=SGDRegressor()
        sgd.fit(x_train,y_train)
        y_test_predict1 = sgd.predict(x_test)
        print('随机梯度 r_square score: ',r2_score(y_test,y_test_predict1))
        print('最小二乘 r_square score: ',r2_score(y_test,y_test_predict))
        red = plt.scatter(np.arange(0,50),sorted(y_test_predict[0:50]),color=(232/255,90/255,104/255))  # arange(起始：末尾)
        blue = plt.scatter(np.arange(0,50),sorted(y_test[0:50]),color=(70/255,147/255,234/255))
        plt.title('成绩预测')
        plt.xlabel('样本编号')
        plt.ylabel('成绩')
        plt.legend([red,blue],['预测值','真实值'])

        buffer = io.BytesIO()
        plt.savefig(buffer,dpi=400,format='PNG',transparent = True)
        img = Image.open(io.BytesIO(buffer.getvalue()))
        plt.close()
        buffer.close()
        return img

    if(type == 'PGE'):sc = str(predict_score(scaleX, lr,cet4,cet6,ur))
    return sc



