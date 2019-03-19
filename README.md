## 算法
让生成的embedding计算得到的相似度矩阵逼近原始相似度矩阵

## 函数
使用时主要关注verse.py中的Verse类，下面简单介绍Verse类的主要方法。
* 初始化函数：\_\_init__(self, instance_number, dimension)  
  * instance_number: 网络中节点数
  * dimension：embedding的维度
* 生成embedding函数：embedding(self, similarity_file, epoches, log=False)
  * similarity_file：相似度文件路径
  * epoches：模型迭代次数，建议设置为3000
  * log：每次迭代是否输出loss，否认不输出
* 保存embedding函数：save_embedding(self, save_file) 
  * save_file：保存文件名
  
## 文件格式
* 相似度文件  
采用标准csv格式，source为源节点id，target为目标节点id，similarity为相似度值。  
示例文件见citeseer_CN.csv，e.g:  

        source,target,similarity
        0,268,0.038461538
        0,295,0.038461538
        0,1175,0.038461538
        ...
* embedding文件  
采用标准csv格式，nodeID为节点id，后面紧跟节点embedding，对应每个维度。  
维度为16的embedding文件，e.g:  

        nodeID,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
        0,-0.12102938,0.08187926,0.09003983,0.14077897,-0.0027368413,0.13024694,-0.079879306,0.10257288,0.10346175,0.076681405,-0.104245506,0.064385615,0.16350543,-0.16906787,-0.09007224,0.106209256
        1,0.13748983,0.031211883,-0.1354264,0.08301413,-0.10816899,0.060328417,-0.15001796,0.0830121,-0.092105374,0.13242689,0.1629433,0.054039776,0.08023161,0.10998872,-0.11312806,0.11284387
        ...

## 使用流程
代码使用python3编写，稍微修改可以用python2运行
* 安装依赖
```
pip install -r requirements.txt
```
* 运行示例main.py
```
# 具体模型调用过程查看main.py内容，结合函数解释理解
python main.py
```

## 论文
https://dl.acm.org/citation.cfm?id=3186120
