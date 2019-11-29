# end2end_ml
端到端机器学习与深度学习

## quick_draw_Autodrawer模型
#### 说明
根据用户的起始笔画,来完成后续的简笔画.
#### 数据集
使用[quick_draw dataset](https://github.com/googlecreativelab/quickdraw-dataset).<br>
#### 模型训练
https://github.com/zhaocc1106/machine_learn/tree/master/NeuralNetworks-tensorflow/RNN/quick_draw<br>
训练完之后通过tensorflowjs_converter --input_format keras model.h5 ./命令将模型转换为tensorflow.ts模型.
#### 效果
https://zhaocc1106.github.io/end2end_ml/quick_draw/auto_draw/<br>
<img src="https://github.com/zhaocc1106/end2end_ml/blob/master/quick_draw/auto_draw/out/autodrawer.gif"  height="400" width="350" alt="autodrawer">
#### 遗留问题
对起始笔画要求比较高,可能是如下原因:<br>
1. 笔画采集和预处理不太满足模型输入<br>
2. 模型本身问题<br>
3. 训练数据不够随机性<br>

后续再继续优化
