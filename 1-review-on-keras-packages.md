
# Keras源码分析(1)：包结构概览

参考：github[https://github.com/keras-team/keras]

Keras在github上的源码包结构如下：
```
|-- docs                      #说明文档
|-- examples                  #应用示例
|-- test                      #测试文件
|-- keras                     #核心源码
      |-- application         #应用实例，如VGG16,RESNET50
      |-- backend             #底层接口，如:tensorflow_backend,theano_backend
      |-- datasets            #数据获取，如boston_housing,mnist
      |-- engine              #网络架构
      |-- layers              #层相关
      |-- legacy              #遗留源码
      |-- preprocessing       #预处理函数
      |-- utils               #实用工具
      |-- wrappers            #scikit-learn封装类
      |-- activations.py      #激活函数
      |-- callbacks.py        #回调函数
      |-- constraints.py      #权重约束，如非零约束等
      |-- initializers.py     #初始化方法
      |-- losses.py           #损失函数
      |-- metrics.py          #度量方法
      |-- models.py           #模型工具
      |-- objectives.py       #目标函数，也就是损失函数，为兼容而保留此文件
      |-- optimizers.py       #优化方法，如SGD，Adam等
      |-- regularizers.py     #正则项，如L1，L2等
```
我们只重点关注以下目录和文件：

1）examples，它是学习keras入门的好地方，所以我们将从这里着手（见下一节），沿着由易到难，由浅入深的方式一步一步阅读分析。

2）application，如果只是想用keras,这里应该是你学习的第二步，从这里你不仅可以学到如何用keras编程，更重要的是你能学到如何架构深度神经网络。在弄清楚几个经典的深度神经网络应用后，你就可以付诸实践，尝试去解决实际应用中的问题了。

3）engine和layers，是Keras的最核心部分，当然也是要重点阅读分析的部分。

4）activations.py，constraints.py，initializers.py，losses.py，metrics.py，optimizers.py，regularizers.py等直接在keras包下的文件，对于这些文件中的内容，可能是理解原理比分析代码更重要。


需要申明的是：

（1）本“Keras源码分析”系列，只是对Keras的主要代码逻辑进行解析，并不是对所有代码逐行注解，因为这容易陷入或纠缠到一些非常具体的细节，甚至是版本细节，从而导致抓小放大；
（2）本“Keras源码分析”系列，也不是对所有类和函数参数进行解释说明，这方面最好和最完整的文档就是Keras的官方文档。
（3）转载请注明出处和链接。