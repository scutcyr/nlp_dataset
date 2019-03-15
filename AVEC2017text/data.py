#!/usr/bin/python3
# South China University of Technology
# Yirong Chen
# mail:eecyryou@mail.scut.edu.cn


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import csv
import collections
import numpy as np
import pandas as pd
import tensorflow as tf
# from google bert
import modeling  
import optimization
import tokenization

# 以下两句代码解决这个警告：2019-01-11 19:58:45.437385: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

################################################################################
# Python version: 3.6
# tensorflow: 1.12.0
# numpy: 1.15.4
# 这是导入AVEC2017抑郁症数据库的text部分的文件
# 数据集路径如下：
'''
/157Dataset/data-chen.yirong/nlpdataset/AVEC_17_18_text

/157Dataset/data-chen.yirong/nlpdataset/AVEC_17_18_text/AVEC2017text
dev_split_Depression_AVEC2017.csv
test_split_Depression_AVEC2017.csv
train_split_Depression_AVEC2017.csv
Readme.txt

/157Dataset/data-chen.yirong/nlpdataset/AVEC_17_18_text/AVEC2017text/DAICWOZ_scripts      # 文件夹里面的每一个文件就是一个谈话内容
## 数据集错误说明
# 陈艺荣，2019年03月01日
# 缺失 342、394、398、460
# 另外451、458、480存在问题，仅有Participant发言  
300_TRANSCRIPT.csv
301_TRANSCRIPT.csv
...

492_TRANSCRIPT.csv

'''

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("data_path", '/157Dataset/data-chen.yirong/nlpdataset/AVEC_17_18_text/AVEC2017text/', "data path to load dataset")

'''
data_path = '/157Dataset/data-chen.yirong/nlpdataset/AVEC_17_18_text/AVEC2017text/'  # 这里可以根据你的数据集存放的位置而进行修改
dev_split = data_path+'dev_split_Depression_AVEC2017.csv'
test_split = data_path+'test_split_Depression_AVEC2017.csv'
train_split = data_path+'train_split_Depression_AVEC2017.csv'
DAICWOZ_scripts = data_path+'DAICWOZ_scripts/'
'''

class InputExample(object):

  def __init__(self, unique_id, text_a, text_b):
    self.unique_id = unique_id
    self.text_a = text_a
    self.text_b = text_b


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids



# 对files列表进行排序
def sort_key(s):
    # 排序关键字匹配
    # 匹配开头数字序号
    if s:
        try:
            c = re.findall('^\d+', s)[0]
        except:
            c = -1
        return int(c)

# 排序函数
def sort_str_list(alist):
    alist.sort(key=sort_key)
    return alist

# 载入dev_split或test_split或train_split的函数
# filename: 带路径的文件名，例如/157Dataset/data-chen.yirong/nlpdataset/AVEC_17_18_text/AVEC2017text/dev_split_Depression_AVEC2017.csv
def loadlabel(filename):
	print('------------------------------------------------------------------->')
	print('载入以下文件：')
	print(filename)
	data = pd.read_csv(filename, encoding = "utf-8", header = 0 )
	print(data)
	return data
 
# 载入某个Participant的对话数据并生成符合BERT输入的对话对
def loaddata(files_csv , Participant_ID, DAICWOZ_scripts):
	# 对于每个属于Participant_ID的id，检查files_csv中是否存在相应的文件
	# 例如id=301，则检查files_csv中是否存在301_TRANSCRIPT.csv这个文件，存在则打开，不存在则报错
	# 数据集中没有394和460，人为添加两个对话数据文件，方便标签
	print('------------------------------------------------------------------->')
	print('载入以下文件：')
	print('Participant_ID是%d的对话内容：' %Participant_ID , Participant_ID)
	data = pd.read_csv(DAICWOZ_scripts+files_csv[Participant_ID-300, 0], encoding = "utf-8", header = 0 )
	#print(data)
	'''
	将data转换为问答对，形式如下：


	'''
	qa_data = [] # 空列表
	unique_id = 0
	# unique_id = Participant_ID*1000   # Participant_ID = unique_id//1000  # unique_id整除1000
	index = 0
	Ellie_STATE = 0
	text_a = ''
	text_b = ''

	Ellie = re.compile('Ellie')
	Participant = re.compile('Participant')
	str_tabtrans = str.maketrans('\t','\000') # 制作翻译表

	for index, line in enumerate(data.values):
		#unique_id = unique_id+1
		temp_str = str(line)
		temp_str.translate(str_tabtrans)

		if Ellie_STATE == 0:   # 读入接下来的句子为text_a
			m = Ellie.search(temp_str)
			if m != None:  # 找到Ellie匹配
				id_e = m.end(0)
				if text_a == '':              
					text_a = tokenization.convert_to_unicode(temp_str[id_e:])
				else:                                             
					text_a = text_a+'. '+tokenization.convert_to_unicode(temp_str[id_e:])
			else:
				Ellie_STATE = 1
				m = Participant.search(temp_str)
				id_p = m.end(0)   #re.search('Participant', temp_str, flags = re.I).end()
				text_b = tokenization.convert_to_unicode(temp_str[id_p:])
		else:
			m = Participant.search(temp_str)
			if m != None:  # 找到Participant匹配
				id_p = m.end(0)
				if text_b == '':
					text_b = tokenization.convert_to_unicode(temp_str[id_p:])
				else:
					text_b = text_b+'. '+tokenization.convert_to_unicode(temp_str[id_p:])
			else:
				Ellie_STATE = 0
				label = str(Participant_ID)#文本对应的情感类别
				print('Ellie:' , text_a)
				print('Participant_%d :' % Participant_ID , text_b)
				qa_data.append(InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))#加入到InputExample列表中
				unique_id = unique_id+1
				text_a = ''
				text_b = ''
				m = Ellie.search(temp_str)
				id_e = m.end(0)
				text_a = tokenization.convert_to_unicode(temp_str[id_e:])


	#print(qa_data)
	qa_len = len(qa_data)
	print(qa_len)
	return Participant_ID, qa_data, qa_len    # 返回的是一个Participant的Participant_ID,所有对话对qa_data，以及对话次数

# 载入train或dev或test数据集
def loaddataset(files_csv , filename, DAICWOZ_scripts):
	dataset = loadlabel(filename)
	print('逐个提取对话数据：')
	print(filename)
	if filename == FLAGS.data_path+'test_split_Depression_AVEC2017.csv':
		dataset_id_label = dataset.loc[:,['participant_ID','Gender'] ]
	else:
		dataset_id_label = dataset.loc[:,['Participant_ID','PHQ8_Binary'] ]
	dataset_matrix = dataset_id_label.as_matrix() # Dataframe指定列转化为矩阵matrix
	
	for Participant_ID in dataset_matrix[:,0]:
		print(Participant_ID)
		loaddata(files_csv , Participant_ID, DAICWOZ_scripts)
	
	return dataset_matrix



# 提取某一对话数据的信息

def load_DAICWOZ_scripts_name(DAICWOZ_scripts):
	files = os.listdir(DAICWOZ_scripts)  # 获得的是1*n的矩阵，print的时候很难看
	files = sort_str_list(files) # 按文件名从300到492进行排序，方便文件读取操作
	files_csv = list(filter(lambda x: x[-4:]=='.csv' , files)) #通过lambda和filter，剔除尾缀不是.csv的文件名
	files_csv = np.reshape(files_csv, (-1,1))  # 将矩阵reshape为1*n的矩阵，print的时候很好看
	print('files_csv:')
	print(files_csv)
	return files_csv

if __name__ == "__main__":

	data_path = FLAGS.data_path
	dev_split = data_path+'dev_split_Depression_AVEC2017.csv'
	test_split = data_path+'test_split_Depression_AVEC2017.csv'
	train_split = data_path+'train_split_Depression_AVEC2017.csv'
	DAICWOZ_scripts = data_path+'DAICWOZ_scripts/'


	files_csv = load_DAICWOZ_scripts_name(DAICWOZ_scripts)

	# 生成Participant_ID
	start_id = 300
	end_id = 492
	step = 1  # 间距为1
	Participant_ID = np.array([id for id in range(start_id, end_id, step)])   # 300 到 492
	print(Participant_ID)

	train_split_data = loadlabel(train_split)
	dev_split_data = loadlabel(dev_split)
	test_split_data = loadlabel(test_split)

	print('测试：')
	train_split_id = [id[2:] for id in train_split_data]
	print(train_split_id)


	loaddata(files_csv , 300, DAICWOZ_scripts)

	# 根据train_split_Depression_AVEC2017逐个提取对话数据
	print('根据train_split_Depression_AVEC2017逐个提取对话数据：')
	print(train_split_data)
	print(len(train_split_data))
	print(train_split_data.columns)
	print(train_split_data.index)
	# print(train_split_data[0:1, 0:1])
	print(train_split_data.loc[:,['Participant_ID','PHQ8_Binary'] ])
	train_id_label = train_split_data.loc[:,['Participant_ID','PHQ8_Binary'] ]
	train_matrix = train_id_label.as_matrix() # Dataframe指定列转化为矩阵matrix
	print(train_matrix)
	print(train_matrix[0][1])
	print(train_matrix[:,0])


	train_matrix = loaddataset(files_csv , train_split, DAICWOZ_scripts) # 'Participant_ID','PHQ8_Binary'
	dev_matrix = loaddataset(files_csv , dev_split, DAICWOZ_scripts)  # 'Participant_ID','PHQ8_Binary'
	test_matrix = loaddataset(files_csv , test_split, DAICWOZ_scripts) # 'participant_ID','Gender'




