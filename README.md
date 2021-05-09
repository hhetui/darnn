# 实验介绍

- ver2        基础LSTM
- ver2_1      改变隐向量长度

- ver3        EN-DN结构

- ver4        单独测试师兄给的CNN-selfattention结构

- ver5        基础 DARNN 模型（回归）
- ver5_1      改变隐向量长度

- ver6        DARNN+self-attention结构
- ver6_1      改变隐向量长度 64
- ver6_2      改变隐向量长度 128
- ver6_3      改变time_step 30
- ver6_4      改变time_step 10

- ver7        在DARNN+selfattention的ENDN中先用self-attention处理数据
- ver8        将DARNN+selfattention的attention部分直接替换为self-attention
- ver9        将DARNN+selfattention的attention部分直接替换为师兄给的CNNself-attention

-ver10        将DARNN+selfattention的attention部分直接替换为师兄给的第二篇CNN的down-sampling部分过拟合增加正则化项
-ver10_1      增加dropout 0.3
-ver10_2      将隐藏层128减少到32

