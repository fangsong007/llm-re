数据集来源：DEIE(数据集): Benchmarking Document-level Event Information Extraction with a Large-scale Chinese News Dataset, LREC-COLING 2024, https://aclanthology.org/2024.lrec-main.410/

该目录用的是顶刊的数据集，并进行数据划分（8：2）：train_split.json和test_split.json

re_bigmodl目录下是基于讯飞星火大模型接口调用实现的新闻文本事件关系识别与抽取算法，基于规则驱动，并引入惩罚机制，构建置信度评分模型；
其中high_re_words.py获取关键词列表；None_knowledge.py是对照实验，没有进行数据增强；Re_opti.py是正式模型；

model_train.py是一个迁移学习的深度模型，需要roberta_zh_L-6-H-768_A-12预训练文件才能运行，由于版权问题在这里没有直接给出，需要者可以去https://github.com/brightmart/roberta_zh 下载，然后放到同级目录下即可。具体包含如下
![image](https://github.com/user-attachments/assets/22751654-2df7-4eb1-b71a-139c7021a522)


训练好后会生成一个event_relation_model文件夹和training_info.txt，然后就可以进行预测predict.py

predictions.txt、predictions1.txt、predictions2.txt分别是在不同Epochs和Batch Size下预测得到的结果（3-2、3-6、10-6）

其余一些py文件是本人的一些模型评估以及可视化的代码，供参考
