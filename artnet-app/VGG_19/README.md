# VGG-19代码上和之前不同的地方

> Download the checkpoint at:  https://drive.google.com/open?id=1C75piZ_YxpFzOQyhif_ERpaWov5PC4gd

- 模型load：load checkpoint（参考已有代码）

- 图像处理：

  - predict函数只能接受PIL image，图片都需要用Image.open(path)打开
  - patch division之后也都需要转为PIL图片（参考已有代码）

- 有9类，对应的snn input output dim都需要修改

- predict函数返回的东西很神奇，一个probability数组，一个label数组，其中probability是从大到小的一些概率，label是每个概率对应的类别（每张图片返回的label数组都不一样）。用

  ```python
              probs, top_labels = predict(patch, model, 9)
              # 创建一个长度为 9 的列表，初始化为 0（因为有 9 个类别）
              prob_vector = [0.0] * 9
  
              # 根据 LABEL_TO_INDEX 的顺序填充概率值
              for label, prob in zip(top_labels, probs):
                  index = LABEL_TO_INDEX[label]
                  prob_vector[index] = prob
  ```

  可以转化为和之前一样的probability vector