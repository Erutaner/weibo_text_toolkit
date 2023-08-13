import json
from weibo_nlp_model_api import ErnieWeiboEmbeddingSentiment, ErnieWeiboEmbeddingSemantics, ErnieWeiboSentiment
from data_cleaning import clean_weibo_comment_data
from data_convert import pie_data_aggregrate
from weibo_visualize import pie_chart, elbow_visualize, polarization_visualize
import pandas as pd
from text2vec import SentenceModel
from weibo_visualize import count_over_days
from weibo_visualize import auto_polarization_visualize
import plotly.graph_objects as go

# weight_path = r"C:\Users\29032\daily_coding\大创\hugging_face_learning\torch_model\CoSENT_Weibo_Multi_BiLSTM_Cls_New_Data"
#
# model = ErnieWeiboSentiment(weight_path=weight_path)
model = ErnieWeiboEmbeddingSemantics()
#
# df = pd.read_csv(r"D:\桌面\武大本科期间文件\大创\系统建设\系统\数据清洗\数据\首次测试_鼠头鸭脖\鼠头鸭脖.csv")
#
# # count_over_days(df,title="疫情放开走势图")
#
# df = clean_weibo_comment_data(df,"微博正文")
#
# text_list = list(df["微博正文"])
#
# output = model.predict(text_list, batch_size=15)
#
# # auto_polarization_visualize(*output,title = "疫情放开观点可视化")
#
# sentence_list, labels = output
#
# # elbow_visualize(embedding_list)
#
# # polarization_json = polarization_visualize(*output, 5, title="鼠头鸭脖", return_dict=True)
#
#
# df_sentiment = pie_data_aggregrate(labels,6)
#
# pie_chart_json = pie_chart(df_sentiment,return_dict=True)
# file_path = r"D:\桌面\武大本科期间文件\大创\系统建设\系统\机器学习\格式数据\鼠头鸭脖饼图六分类.json"
# with open(file_path, 'w') as file:
#     json.dump(pie_chart_json,file)




