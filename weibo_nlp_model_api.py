"""
Author: Erutaner
Date: 2023.08.08
"""
import torch
import transformers
from transformers import AutoTokenizer, AutoModel
from torch import cuda
import numpy as np
from text2vec import SentenceModel

class WeiboCosentClass(torch.nn.Module):
    '''使用shibing624的预训练模型做encoder，在两万条微博评论数据上针对情感分类任务进行了微调'''
    def __init__(self):
        super(WeiboCosentClass, self).__init__()
        self.encoder = AutoModel.from_pretrained("shibing624/text2vec-base-chinese-paraphrase")
        # 第一个768是bert输出的768，第二个是指定的LSTM的hidden size
        self.bilstm = torch.nn.LSTM(768, 768, batch_first = True, bidirectional = True, num_layers = 5)
        # 使用30%的概率随机丢弃，防止过拟合
        self.dropout = torch.nn.Dropout(0.3)
        self.relu = torch.nn.ReLU()
        # 768*2是因为双向的LSTM后面要在hidden维进行两个方向的拼接
        self.fc = torch.nn.Linear(768*2,512)
        self.classifier = torch.nn.Linear(512, 6)
        # # 定义权重初始化函数，防止训练时发生梯度消失
        # def init_weights(m):
        #     if isinstance(m, torch.nn.LSTM):
        #         for name, param in m.named_parameters():
        #             if 'weight_ih' in name:
        #                 torch.nn.init.xavier_uniform_(param.data)
        #             elif 'weight_hh' in name:
        #                 torch.nn.init.orthogonal_(param.data)
        #             elif 'bias' in name:
        #                 param.data.fill_(0)
        #     elif isinstance(m, torch.nn.Linear):
        #         torch.nn.init.xavier_uniform_(m.weight)
        #         m.bias.data.fill_(0.01)
        # # 应用权重初始化到bilstm, fc和classifier层
        # self.bilstm.apply(init_weights)
        # self.fc.apply(init_weights)
        # self.classifier.apply(init_weights)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_ernie = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # output_ernie是个二元组，取出第一个元素，是个(batch_size, sequence_length, hidden_dim)的三维张量，里面有我们要的隐状态
        output_1 = output_ernie[0]
        # 取出双向深层LSTM的最深的正反向末尾隐状态，进行拼接
        bi_output, (hn, cn) = self.bilstm(output_1)
        cat = torch.cat((hn[-2,:,:],hn[-1,:,:]),dim=1)
        rel = self.relu(cat)
        dense_1 = self.fc(rel)
        drop = self.dropout(dense_1)
        output = self.classifier(drop)
        # 这里返回的并不是最终标签或概率，而是logits，还需经softmax处理
        return output

class EmbeddingModel(torch.nn.Module):
    '''本质上是上面的情感分类模型去掉分类头，可做句嵌入，将句子转化为768维向量，情感相近的句子对应的向量具有相近的欧氏距离，向量可用于下游情感分析任务'''
    def __init__(self, original_model):
        super(EmbeddingModel, self).__init__()
        self.emb_layer = original_model.encoder

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_ernie = self.emb_layer(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_ernie[0]
        batched_cls = hidden_state[:,0]
        return batched_cls




class ErnieWeiboSentiment():
    '''
        使用时请关闭VPN并联网，否则无法获取tokenizer
        情感分析API，实例化时需要传入模型权重的本地路径，权重文件下载地址见readme。
        注意传入地址时要使用r""字符串格式，否则容易出现格式错误。
        实例化此类后，只需在实例对象上调用.predict方法，传入文本列表，便可进行情感分析。
        支持批量读入，默认批量大小为32，意思是一次读取32条文本。
        支持cpu运行，不需要电脑装有gpu。
        输入数据为字符串列表，其中的每个元素为一条文本。
        返回数据为一个二元组，二元组的第一个元素是传入的文本列表，第二个元素为情感的数字标签列表（一维列表），数字标签与文本一一对应。
        数字标签对应的情感可通过列表：["正向","中性","吐槽","伤心","恐惧","愤怒"]进行获取，例如，标签0对应情感“正向”，标签1对应“中性”。
    '''
    def __init__(self, weight_path=None, encoder_path="shibing624/text2vec-base-chinese-paraphrase"):
        self.model = WeiboCosentClass()
        self.model.load_state_dict(torch.load(weight_path))
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_path, truncation=True,
                                                       padding=True)
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()

    def predict(self, input_text_list, batch_size=32):
        results = []
        num_batches = (len(input_text_list) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(input_text_list))
            current_batch = input_text_list[start_idx:end_idx]

            inputs = self.tokenizer(
                current_batch,
                add_special_tokens=True,
                max_length=256,
                padding='max_length',
                truncation=True,
                return_token_type_ids=True,
                return_tensors='pt'
            )

            for key, val in inputs.items():
                inputs[key] = val.to(self.device)

            with torch.no_grad():
                output_logits = self.model(**inputs)
                _, pred_val = torch.max(output_logits, dim=1)

            if self.device != "cpu":
                pred_val = pred_val.cpu().numpy().tolist()
            else:
                pred_val = pred_val.numpy().tolist()

            results.extend(pred_val)

        return input_text_list, results

class ErnieWeiboEmbeddingSemantics():
    '''
    本模型是shibing624的工作
    经过测试，本模型嵌入的句向量对语义敏感而对文本情感不敏感
    '''

    def __init__(self,weight_path = "shibing624/text2vec-base-chinese-paraphrase"):
        self.model = SentenceModel(weight_path)
    def encode(self, input_text_list, batch_size = 32):
        embedding_list =  self.model.encode(input_text_list,batch_size)
        return input_text_list, embedding_list


class ErnieWeiboEmbeddingSentiment():
    '''
    本模型嵌入的句向量对情感敏感而语义表征能力欠佳，建议用于下游情感分析任务，其他下游nlp任务请使用ErnieWeiboEmbeddingSemantics
    使用时请关闭VPN并联网，否则无法获取tokenizer。
    本类为微博文本句嵌入API，在实例化时只需传入模型权重的本地路径，权重文件下载地址见readme。
    注意传入地址时要使用r""字符串格式，否则容易出现格式错误。
    在实例对象上调用embed方法，传入需要进行句嵌入的文本列表即可使用，列表中每个元素为一个博文或一个评论（字符串—）。
    支持批量读入，默认值为32，意思是一次读取32条文本，如果跑不动可以调小这个值（得是正整数）
    返回值为一个二元组，元组的第一个元素为传入的文本列表（原样返回），元组的第二个元素为np.ndarray，数据类型np.float64，即一个二维矩阵。
    二维矩阵每一行对应传入文本的句嵌入向量
    '''
    def __init__(self, weight_path=None):
        self.original_model = WeiboCosentClass()
        self.original_model.load_state_dict(torch.load(weight_path))
        self.model = EmbeddingModel(self.original_model)
        self.tokenizer = AutoTokenizer.from_pretrained("shibing624/text2vec-base-chinese-paraphrase", truncation=True,
                                                       padding=True)
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()

    def encode(self, input_text_list, batch_size=32):

        num_batches = (len(input_text_list) + batch_size - 1) // batch_size
        #  np.float64支持高精度计算
        embedding_list = np.empty((0, 768), np.float64)
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(input_text_list))
            current_batch = input_text_list[start_idx:end_idx]

            inputs = self.tokenizer(
                current_batch,
                add_special_tokens=True,
                max_length=256,
                padding='max_length',
                truncation=True,
                return_token_type_ids=True,
                return_tensors='pt'
            )

            for key, val in inputs.items():
                inputs[key] = val.to(self.device)

            with torch.no_grad():
                output_logits = self.model(**inputs)

            if self.device != "cpu":
                pred_val = output_logits.cpu().numpy()
            else:
                pred_val = output_logits.numpy()

            embedding_list = np.vstack((embedding_list, pred_val))

        return input_text_list, embedding_list



device = "cuda" if cuda.is_available() else 'cpu'
if device == "cuda":
    print(f"Your current device is: {device}, which means you will run the model on gpu.")
else:
    print(f"Your current device is: {device}, which means you will run the model on cpu.")



