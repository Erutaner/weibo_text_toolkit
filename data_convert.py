"""
Author: Erutaner
Date: 2023.08.08
"""
import pandas as pd
label2sentiment = ["正向","中性","吐槽","伤心","恐惧","愤怒"]

def pie_data_aggregrate(label_list,class_num = 6):
    '''
    传入参数第一个是情感数字标签列表，即ErnieWeiboSentiment类的实例对象调用predict方法返回的元组的第二个元素
    返回dataframe有两列，sentiment列和count列，记录每种sentiment的数量
    '''
    if class_num == 6:
        new_label_list = []
        for j in label_list:
            new_label_list.append(label2sentiment[j])
    elif class_num == 3:
        new_label_list = []
        for j in label_list:
            if j>=2 and j<=5:
                new_label_list.append("负向")
            else:
                new_label_list.append(label2sentiment[j])
    else:
        raise ValueError("The argument class_num must be 3 or 6.")
    df = pd.DataFrame(new_label_list,columns=["sentiment"])
    df["count"] = 1
    df = df.groupby("sentiment").sum().reset_index()
    return df
