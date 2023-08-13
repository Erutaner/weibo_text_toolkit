"""
Author: Erutaner
Date: 2023.08.08
"""
import pandas as pd
import re

def clean_weibo_comment_data(df, column_name, is_training = False):
    '''
    与爬虫配套的数据清洗函数，如果用于is_traning = True将会去除70%新闻文本。
    传入一个pandas dataframe，column_name是一个字符串，对应文本列列名。
    返回一个dataframe，字段（列）跟传入的dataframe一致，但文本列已被清洗
    '''
    # 去除新闻描述型文本（带有【字符的文本）、带有书名号及网页链接的文本（存在不与其他数字相关的0）
    df = df[~df[column_name].str.contains('\u3010|\u330A|(?<!\d)\u004F(?!\d)')]
    # 去除文本中的emoji和##包裹的话题名称，去除视频引用，去除毫无意义的英文?（该问号是爬虫爬下来的）
    combined_pattern = "[\U00002700-\U000027BF\U00002600-\U000026FF\U0001F000-\U0001FBF9\U00002300-\U000023FF\U0000FFF0-\U0000FFFF\U000025A0-\U000025FF\U00002B00-\U00002BFF]+|#[^#]*#|L.*的微博视频|\?|\u300E.*?\u300F|\s|\u200b"
    df.loc[:, column_name] = df[column_name].str.replace(combined_pattern, '', regex=True)
    # 去除文本中的//@用户名:发言格式，去除文本中的[]表情残余，去除文本中的网址
    df.loc[:,column_name] = df[column_name].str.replace(r"//@.*?:|\[.*?\]|http.*?(\s|$)", "", regex=True)
    # 去除上述处理后参与的@用户名，到空格截止
    df.loc[:, column_name] = df[column_name].str.replace(r"@.*?\s", "", regex=True)
    # 清洗训练语料时仅保留30%的含有《的行，这是因为这些行大部分都是新闻文本，数量和字数都极大，不利于模型学习
    if is_training:
        pattern = "《"
        # 找出含有《的所有行
        mask = df[column_name].str.contains(pattern)
        # 对这些行随机抽样，抽样比例为70%
        to_remove = df[mask].sample(frac=0.7, random_state=42)
        # 去除这些行
        df = df.drop(to_remove.index)
    # 去除较长博文末尾存在的收起d
    # 去掉收起d
    df.loc[:, column_name] = df[column_name].str.replace('收起d', '', regex=False)
    # 去除文本末尾存在的2地名
    df.loc[:, column_name] = df[column_name].str.replace('2\D{0,5}$', '', regex=True)
    df.loc[:, column_name] = df[column_name].str.replace('2.{0,5}·.*$', '', regex=True)
    # 去掉清洗后评论字段为空串的行
    df = df[df[column_name].str.strip() != '']

    # 去除重复行，以免小广告刷屏
    df = df.drop_duplicates(subset=column_name)
    return df