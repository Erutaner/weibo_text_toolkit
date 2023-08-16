"""
Author: Erutaner
Date: 2023.08.08
"""
import plotly.express as px
import plotly.io as pio
import json
import pandas as pd
import plotly.graph_objects as go
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from kneed import KneeLocator
import numpy as np
from ml_algorithms import CosineKMeans
from polarization_metric import compute_entropy, compute_discrepancy

def pie_chart(df,values = "count", names = "sentiment", title = "情感分布",return_dict=False):
    '''
    绘制饼图，展示各个情感文本的占比
    第一个参数传入的是pie_data_aggregrate返回的dataframe, 有两列，一列是sentiment列，一列是count列
    return_dict为True时，返回一个json格式的字符串，需要用json.dump转化为json格式
    '''
    fig = px.pie(df,values=values,names=names)
    fig.show()

    if return_dict:
        json_data = pio.to_json(fig)
        json_data = json.loads(json_data)
        return json_data


def count_over_days(df,time_column = "时间", title = "Count over Time",return_dict = False):
    '''
    绘制舆情热度变化曲线，time_column和title指定存储时间的列和图的标题
    可以直接传入数据清洗完成后的dataframe，传入储存时间数据的列名和图表的题目
    return_dict为True时，返回一个json格式的字符串，需要用json.dump转化为json格式
    '''
    df[time_column] = pd.to_datetime(df[time_column])

    # 提取日期部分
    df["date"] = df[time_column].dt.date

    # 统计每一天的记录数量
    result = df["date"].value_counts()

    # 转换为 DataFrame 并重命名列
    result_df = result.reset_index()
    result_df.columns = ['Date', 'Count']

    # 按照日期列排序
    result_df = result_df.sort_values('Date')
    fig = px.line(result_df, x='Date', y="Count")
    fig.update_layout(
        title=title,
    )
    fig.show()
    if return_dict:
        json_data = pio.to_json(fig)
        json_data = json.loads(json_data)
        return json_data



def elbow_visualize(embedding_list, start=1, end=10):
    '''
    手肘法绘图API，用于consine-means搜参，所谓的cosine-means其实就是k-means对归一化向量进行处理，对语义向量的聚类更准。
    start和end指定搜参范围（包含端点两侧），默认是闭区间[1,10]
    embedding_list传入文本列表进行句嵌入后的二维numpy矩阵（ndarray）
    '''
    sse = []
    k_range = range(start, end+1)
    for k in k_range:
        kmeans = CosineKMeans(n_clusters=k)
        kmeans.fit(embedding_list)
        sse.append(kmeans.inertia_)

    elbow_df = pd.DataFrame({'Number of Clusters': k_range, 'SSE': sse})

    # 使用Plotly Express绘制图像
    fig = px.line(elbow_df, x='Number of Clusters', y='SSE', title='Elbow Method for Optimal Number of Clusters')
    fig.show()


def polarization_visualize(text_list, embedding_list, n_clusters, title = "Polarization Visualization", return_dict = False):
    '''
    需要配合elbow_visualize进行使用
    text_list为文本列表
    embedding_list为嵌入后的numpy二维矩阵（ndarray），可通过本项目模型得到
    n_cluster为指定聚类数量，通过手肘法确认
    return_dict为True时，返回一个json格式的字符串，需要用json.dump转化为json格式
    '''
    max_line_length = 40  # 每行的最大长度
    # 添加html换行符，以避免文本单行显示
    text_list_multi_line = ['<br>'.join([text[i:i + max_line_length] for i in range(0, len(text), max_line_length)]) for
                            text in text_list]
    # 使用k-means进行聚类
    kmeans = CosineKMeans(n_clusters=n_clusters)
    kmeans.fit(embedding_list)
    labels = kmeans.labels_

    # 使用PCA降维到3维
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(embedding_list)

    # 创建一个DataFrame来存储降维后的数据以及聚类标签
    reduced_df = pd.DataFrame(reduced_data, columns=['Component 1', 'Component 2', 'Component 3'])
    reduced_df['Cluster'] = labels

    # 创建一个3D散点图
    fig = go.Figure(data=[go.Scatter3d(x=reduced_df['Component 1'],
                                       y=reduced_df['Component 2'],
                                       z=reduced_df['Component 3'],
                                       mode='markers',
                                       marker=dict(size=2,
                                                   color=reduced_df['Cluster'],
                                                   colorscale='Viridis',
                                                   colorbar=dict(title='')),  # 定义光谱颜色映射
                                       text=text_list_multi_line,  # 添加文本标签
                                       hovertemplate='Cluster: %{marker.color}<br>Text: %{text}}<extra></extra>'
                                       # 定义悬停文本

                                       )])
    fig.update_layout(title=title)

    fig.show()
    if return_dict:
        json_data = pio.to_json(fig)
        json_data = json.loads(json_data)
        return json_data


def auto_polarization_visualize(text_list, embedding_list, title="Polarization Visualization", return_dict=False):
    '''
    作三维散点图，可通过鼠标悬停查看观点簇中的文本
    基于kneedle算法自动对cosine-k-means搜参，无需指定聚类数
    text_list为文本列表
    embedding_list为嵌入后的numpy二维矩阵（ndarray），可通过本项目模型得到
    title参数为作图标题
    可选return_dict，返回的是个json格式的字符串，需要用json dump写入
    '''
    max_line_length = 40  # 每行的最大长度
    # 添加html换行符，以避免文本单行显示
    text_list_multi_line = ['<br>'.join([text[i:i + max_line_length] for i in range(0, len(text), max_line_length)]) for
                            text in text_list]
    # 自动搜参
    x = range(1, 11)
    y = []
    for k in x:
        kmeans = CosineKMeans(n_clusters=k)
        kmeans.fit(embedding_list)
        y.append(kmeans.inertia_)
    kn = KneeLocator(x, y, curve="convex", direction="decreasing", interp_method="polynomial")
    best_k = kn.knee if kn.knee is not None else 4

    # 使用cosine-k-means进行聚类
    kmeans = CosineKMeans(n_clusters=best_k)
    kmeans.fit(embedding_list)
    labels = kmeans.labels_

    # 使用PCA降维到3维
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(embedding_list)

    # 创建一个DataFrame来存储降维后的数据以及聚类标签
    reduced_df = pd.DataFrame(reduced_data, columns=['Component 1', 'Component 2', 'Component 3'])
    reduced_df['Cluster'] = labels

    # 创建一个3D散点图
    fig = go.Figure(data=[go.Scatter3d(x=reduced_df['Component 1'],
                                       y=reduced_df['Component 2'],
                                       z=reduced_df['Component 3'],
                                       mode='markers',
                                       marker=dict(size=2,
                                                   color=reduced_df['Cluster'],
                                                   colorscale='Viridis',
                                                   colorbar=dict(title='', tickvals=list(range(best_k)),
                                                                 ticktext=list(range(best_k)))),  # 定义光谱颜色映射
                                       text=text_list_multi_line,  # 添加文本标签
                                       hovertemplate='Cluster: %{marker.color}<br>Text: %{text}}<extra></extra>'
                                       # 定义悬停文本

                                       )])
    fig.update_layout(title=title)

    fig.show()
    if return_dict:
        json_data = pio.to_json(fig)
        json_data = json.loads(json_data)
        return json_data


def auto_polarization_over_time_visualize(text_list, embedding_list, dates, title="Polarization Visualization",
                                return_dict=False):
    '''
    text_list为文本列表
    embedding_list为嵌入后的numpy二维矩阵（ndarray），可通过本项目模型得到
    dates为与text_list文本顺序相同的dataframe的日期列，经过clean_weibo_comment_data进行数据清洗后直接传入时间列
    自动搜参，基于cosine-kmeans算法，见mlalgorithms.py，
    return_dict为True时，返回一个json格式的字符串，需要用json.dump转化为json格式
    '''
    dates = dates.reset_index(drop=True)
    max_line_length = 40  # 每行的最大长度
    # 将日期转换为日期类型
    dates = pd.to_datetime(dates)
    unique_dates = dates.dt.date.unique()

    all_figs_data = []

    for date in unique_dates:
        indices = dates[dates.dt.date == date].index
        current_texts = [text_list[i] for i in indices]
        current_embeddings = embedding_list[indices]


        text_list_multi_line = ['<br>'.join([text[i:i + max_line_length] for i in range(0, len(text), max_line_length)])
                                for text in current_texts]

        # 自动搜参
        x = range(1, 11)
        y = []
        for k in x:
            kmeans = CosineKMeans(n_clusters=k)
            kmeans.fit(current_embeddings)
            y.append(kmeans.inertia_)
        kn = KneeLocator(x, y, curve="convex", direction="decreasing", interp_method="polynomial")
        best_k = kn.knee if kn.knee is not None else 4

        # 使用cosine k-means进行聚类
        kmeans = CosineKMeans(n_clusters=best_k)
        kmeans.fit(current_embeddings)
        labels = kmeans.labels_

        # 使用PCA降维到3维
        pca = PCA(n_components=3)
        reduced_data = pca.fit_transform(current_embeddings)

        # 创建一个DataFrame来存储降维后的数据以及聚类标签
        reduced_df = pd.DataFrame(reduced_data, columns=['Component 1', 'Component 2', 'Component 3'])
        reduced_df['Cluster'] = labels

        # 创建一个3D散点图
        fig = go.Figure(data=[go.Scatter3d(x=reduced_df['Component 1'],
                                           y=reduced_df['Component 2'],
                                           z=reduced_df['Component 3'],
                                           mode='markers',
                                           marker=dict(size=2,
                                                       color=reduced_df['Cluster'],
                                                       colorscale='Viridis',
                                                       colorbar=dict(title='', tickvals=list(range(best_k)),
                                                                     ticktext=list(range(best_k)))),  # 定义光谱颜色映射
                                           text=text_list_multi_line,  # 添加文本标签
                                           hovertemplate='Cluster: %{marker.color}<br>Text: %{text}}<extra></extra>'
                                           # 定义悬停文本

                                           )])
        fig.update_layout(title=title)


        all_figs_data.append(fig.data[0])

    # 创建主图并添加滑动条
    steps = []
    for i, date in enumerate(unique_dates):
        step = dict(
            method="restyle",
            args=["visible", [False] * len(unique_dates)],
            label=str(date)
        )
        step["args"][1][i] = True
        steps.append(step)

    sliders = [dict(
        active=0,
        yanchor="top",
        xanchor="left",
        currentvalue=dict(
            font=dict(size=20),
            prefix="Date:",
            visible=True,
            xanchor="right"
        ),
        pad=dict(b=10, t=50),
        len=0.9,
        x=0.1,
        y=0,
        steps=steps
    )]

    layout = go.Layout(sliders=sliders)

    fig = go.Figure(data=all_figs_data, layout=layout)
    fig.update_layout(title=title)
    fig.show()

    if return_dict:
        json_data = pio.to_json(fig)
        json_data = json.loads(json_data)
        return json_data


def kneedle_cosine_kmeans(embedding_list,start = 1, end = 11):
    '''
    :param embedding_list: 传入句嵌入后得到的embedding_list
    :param start: 传入搜参的聚类数起始值
    :param end: 传入搜参的聚类数结束值
    '''
    x = range(start,end+1)
    y = []
    for k in x:
        kmeans = CosineKMeans(n_clusters = k)
        kmeans.fit(embedding_list)
        y.append(kmeans.inertia_)
    kn = KneeLocator(x,y,curve = "convex",direction = "decreasing",interp_method = "polynomial")
    best_k = kn.knee if kn.knee is not None else 4

    # 使用k-means进行聚类
    kmeans = CosineKMeans(n_clusters=best_k)
    kmeans.fit(embedding_list)
    return kmeans


def polar_degree_over_time(embedding_list, date_series, metric = compute_discrepancy, title="Polarization Over Time", return_dict=False):
    '''
    :param embedding_list: 为嵌入后的numpy二维矩阵（ndarray），可通过本项目模型得到
    :param date_series: 与embedding_list顺序相同的dataframe的日期列，经过clean_weibo_comment_data进行数据清洗后直接传入时间列
    :param title: 绘制图表的标题
    :param return_dict: 设置为True则返回一个json格式的字符串，需要用json.dump转化为json格式
    '''
    # 重置索引
    date_series = date_series.reset_index(drop=True)

    # 将object类型的date_series转换为datetime类型
    date_series = pd.to_datetime(date_series)

    # 确保embedding_list和date_series长度匹配
    assert len(embedding_list) == len(date_series), "Mismatched length between embedding list and date series."

    # 创建一个DataFrame，并按日期排序
    combined_df = pd.DataFrame({
        'Date': date_series,
        'Embeddings': list(range(len(embedding_list)))  # 这里我们只是存储索引，避免存储嵌入向量本身
    }).sort_values(by='Date')

    # 以天为单位对embedding_list进行分组
    unique_dates = combined_df['Date'].dt.date.unique()
    polarization_values = []

    for date in unique_dates:
        current_indices = combined_df[combined_df['Date'].dt.date == date]['Embeddings'].tolist()
        current_embeddings = embedding_list[current_indices]

        # 获取最佳的CosineKMeans模型
        kmeans = kneedle_cosine_kmeans(current_embeddings)

        # 计算该日期的极化度量值
        discrepancy_value = metric(current_embeddings, kmeans)
        polarization_values.append(discrepancy_value)

    # 将结果保存到一个DataFrame中
    result_df = pd.DataFrame({
        'Date': unique_dates,
        'Polarization Degree': polarization_values
    })

    # 使用plotly.express绘制线图
    fig = px.line(result_df, x='Date', y="Polarization Degree")
    fig.update_layout(title=title)
    fig.show()

    if return_dict:
        json_data = pio.to_json(fig)
        json_data = json.loads(json_data)
        return json_data
