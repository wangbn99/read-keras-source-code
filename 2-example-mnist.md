# Keras源码分析(2)：入门示例

文件：keras/examples/mnist_cnn.py

在这里我们选择mnist_cnn作为入门示例。这是一个非常简单的深度卷积神经网络，它运行在MNIST数据集上，在12轮训练之后能达到99.25%的精度。其源码及解释如下：

这很简单，没什么好说的
```sh
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
```
定义模型的超参数: 批大小、标签类别数量和训练轮数。
有些人对这几个概念比较模糊，所以顺便解释一下：批量大小指的是一次训练的样本数目, 它将影响到模型的优化程度和训练速度。当一个完整的数据集通过了神经网络一次并且返回了一次，就称为一个epoch。还有一个概念就是迭代（iteration），迭代是batch需要完成一个epoch的次数。打个比方,一个数据集有2000个训练样本，将这2000个样本分成大小为500的batch，那么完成一个epoch就需要4个iteration。
```sh
batch_size = 128
num_classes = 10
epochs = 12
```
定义输入图像的维度尺寸，要与数据集中图像的大小要一致。
```sh
img_rows, img_cols = 28, 28
```
获取数据集，至于如何获取，将在后面接下来的两节进行分析。mnist.load_data返回的是这样一个形式的元组：（训练集，测试集），（训练集标签，测试集标签）。
```sh
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```
重新格式化数据集的维度。对于图像文件，有的是chanel在前（channels_first），即：(img_chanels, img_rows, img_cols)，有的是chanel在后（channels_last），即 (img_rows, img_cols, img_chanels)。
```sh
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
```
把数据转换成浮点类型和归一化（Normalization），即将每个数据单元统一映射到[0,1]区间上。如果是有量纲表达式的数据，也要把它们变换为无量纲表达式，成为纯量。经过归一化处理的数据，所有特征都处于同一数量级，可以消除指标之间的量纲和量纲单位的影响，以防止某些特征指标占优。
```sh
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
```
采用one-hot编码处理类别标签，即将一个类别向量转换成二分类矩阵。例如：
    array([0, 2, 1, 2, 0])有3个类别 {0, 1, 2}
    转换后是这个样子：
```sh
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
```
```sh
matricesy_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
````
以下只做了简单注释，因为每个部分都是重点，所以后续要分单独章节阅读分析。
```sh
#new一个Sequential模型 
model = Sequential()  
# 增加层
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# 配置模型
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# 用数据训练模型
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
# 模型评估
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```