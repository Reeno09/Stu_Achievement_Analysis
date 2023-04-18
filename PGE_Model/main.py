import pandas as pd
import matplotlib.pyplot as plt
import sys
import Heatmap
import Score_Count
import Relation
import Worst_to_Best

if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    df = pd.read_csv("/Users/reeno/Documents/Achievement_Analysis/Score/Admission_Predict_Ver2.3.csv", sep =",")
    dfPGE = pd.read_csv("/Users/reeno/Documents/Achievement_Analysis/Score/WPGE_Ver1.32.csv", sep =",")
    dfSA = pd.read_csv('/Users/reeno/Documents/Achievement_Analysis/score/Admission_Predict.csv',sep=',')
    dfPR = pd.read_csv('/Users/reeno/Documents/Achievement_Analysis/score/WSDPR_Ver1.1.csv',sep=',')
    print('There are ', len(df.columns), 'columns')
    for c in df.columns:
        sys.stdout.write(str(c) + ', ')
    #df = df.rename(columns={'KS':'选择深造','WPGE':'选择考研','WWORK':'选择工作','WSA':'选择出国','WPR':'选择保研','WSD':'选择直博','Straight2Doctorate':'直博率','Keep_PGE':'二战率',' Postgraduate_Recommendation':'保研率','Pass':'是否通过考研','SL':'院线','CL':'国家线','Score':'考研总分','Serial No.':'序号','Chance of Admit ': '深造概率','University Rating':'学校等级','SOP':'自身意愿','LOR ':'推荐信','CGPA':'加权平均分','Research':'研习经历','MTH':'数学分数','ENG':'英语分数','Paper':'学术产物','DV2CL':'超国线分差','DV2SL':'超院线分差'})
    df.info()
    #Relation.Relation_btw_SUC_and_UR(dfPR)
    #Heatmap.Draw_HeatMap(dfPR)
    #Worst_to_Best.Draw_MTH_W2B(df)
    #Score_Count.Draw_Score(df)
    #Relation.Relation_btw_CGPA_and_Score(df)
    #Relation.Relation_btw_Math_and_Score(df)
    #Relation.Relation_btw_CET4_and_ENGscore(df)
    #Relation.Relation_btw_UR_and_Pass(df)
    #Relation.Relation_btw_SOP_and_CGPA(df)
    #from Linear_Regression import Linear_Regression
    #Linear_Regression(dfPGE,'dPGE')
    #from Random_Forest_Classifier_Predict import Random_Forest_Classifier
    #Random_Forest_Classifier(dfPGE,'PGE')



