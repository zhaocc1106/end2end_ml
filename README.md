# end2end_ml
end-to-end machine learning\deep learning model.

## quick_draw_Autodrawer模型
### 说明
根据用户的起始笔画,来完成后续的简笔画.
### 数据集
使用[quick_draw dataset](https://github.com/googlecreativelab/quickdraw-dataset).<br>
### 模型训练
https://github.com/zhaocc1106/machine_learn/tree/master/NeuralNetworks-tensorflow/RNN/quick_draw<br>
训练完之后通过tensorflowjs_converter --input_format keras model.h5 ./命令将模型转换为tensorflow.ts模型.
### 效果
https://zhaocc1106.github.io/end2end_ml/quick_draw/auto_draw/<br>
<img src="https://github.com/zhaocc1106/end2end_ml/blob/master/quick_draw/auto_draw/out/autodrawer.gif"  height="600" width="500" alt="autodrawer">

