import pandas as pd
from tqdm import tqdm
import jieba
import matplotlib.pyplot as plt

def GetText(file_path):
    '''
    数据读取函数
    传入txt文件绝对路径
    返回text列表
    '''
    f=open(file_path,'r',encoding='utf8')
    data=[]
    for line in tqdm(f.readlines()):
        text=''
        message=line.split()
        for each in message[2:-8]:
            text+=' '+each
        data.append(text)
    f.close()
    return data

class Tokenizer:
    '''
    对自然语言进行编码
    '''
    def __init__(self,texts,coding='c',PAD=0):
        '''
        输入将要需要操作的文本（一个字符串的列表）
        这里需要完成词典的构建（即汉字到正整数的唯一映射的确定）
        注意构建词典 一是要根据coding来选择按词构建（coding='w')，还是按字构建，默认按字构建；PAD 默认为0。
        '''
        self._texts=texts
        self._coding=coding
        self._PAD=PAD
        self._dic={}
        self._codes={PAD,}
        self._dic.update({'PAD':PAD})
        code=0
        for text in tqdm(texts,desc='initing dictionary...'):
            if coding=='w':
                text=jieba.lcut(text)
            for word in text:
                if word not in self._dic:
                    while code in self._codes:
                        code+=1
                    self._dic.update({word:code})
                    self._codes.add(code)
    def tokenize(self,sentence):
        '''
        输入一句话，返回分词(字)后的字符列表(list_of_chars)
        '''
        res=[]
        for each in tqdm(jieba.cut(sentence),desc='tokenizing...'):
            res.append(each)
        return res

    def encode(self,list_of_chars):
        '''
        输入字符(字或者词）的字符列表，返回转换后的数字列表 (tokens)
        '''
        tokens=[] 
        for word in list_of_chars:
            tokens.append(self._dic[word])
        return tokens

    def trim(self,tokens,seq_len):
        '''
        输入数字列表tokens，整理数字列表的长度。不足seq_len的 部分用PAD补足，超过的部分截断
        '''
        if len(tokens)>=seq_len:
            return tokens[:seq_len]
        else:
            return tokens.extend([self._PAD for i in range(len(tokens)-seq_len)])

    def decode(self,tokens):
        '''
        将模型输出的数字列表翻译回句子。如果有PAD，输出'[PAD]'
        '''
        res=''
        dic_list=[self._PAD for i in range(len(self._dic))]
        for each in self._dic:
            dic_list[self._dic[each]]=each
        for each in tokens:
            if each==self._PAD:
                res+='[PAD]'
            else:
                res+=dic_list[each]
        return res

    def encode_all(self,seq_len):
        '''
        返回所有文本(chars)的长度为seq_len的tokens
        '''
        codes=[[] for i in range(len(self._texts))]
        i=0
        for text in tqdm(self._texts,desc='encoding all...'):
            if self._coding=='w':
                text=jieba.lcut(text)
            for word in text:
                codes[i].append(self._dic[word])
            i+=1
        for code in tqdm(codes,desc='triming...'):
            if len(code)<seq_len:
                code.extend([self._PAD for i in range(seq_len-len(code))])
            else:
                code=code[:seq_len]
        return codes
    def output_distribution(self):
        '''
        输出编码分布
        '''
        xy={}
        for text in self._texts:
            if self._coding=='w':
                text=jieba.lcut(text)
            length=len(text)
            if length in xy:
                xy[length]+=1
            else:
                xy.update({length:1})
        x=sorted(xy.keys())
        y=[xy[key] for key in tqdm(x,desc='loading y...')]
        plt.bar(x,y)
        plt.xlabel('length')
        plt.ylabel('distribution')
        plt.show()

def main():
    '''
    test
    '''
    file_path='D:\Project\Python\week5Tokenizer\\final_none_duplicate0.txt'
    texts=GetText(file_path)
    token=Tokenizer(texts)
    token.output_distribution()

if __name__=='__main__': main()