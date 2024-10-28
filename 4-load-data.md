# Keras源码分析(4)：数据获取

文件：keras/datasets/mnist.py

这个源码可以看作是前篇utils.data_utils.get_file的一个使用案例。
```sh
def load_data(path='mnist.npz'):
    path = get_file(path,
        origin='https://s3.amazonaws.com/img-datasets/mnist.npz',
        file_hash='8a61469f7ea1b51cbae51d4f78837e45')
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)
```
在这个load_data函数中，它简单直接地hard code了数据集文件的URL和文件hash校验值作为utils.data_utils.get_file的参数，进行数据集下载，并得到下载后缓存于本地的完整的文件路径path, 然后利用numpy.load载入数据到内存并返回一个dict，其中包含四项：x_train, y_train，x_test, y_test，它们对应的值是numpy数组。

在你的home目录下，你可以找到这个下载的数据集文件，具体路径是:~/.keras/datasets/mnist.npz，它是一个zip压缩文件，打开它你会发现它包含4个numpy数组序列化存储文件：x_train.npy, y_train.npy，x_test.npy, y_test.npy, 这也是为什么np.load返回包含相应4个键值的原因。

MNIST是一个包含60,000个训练图像和10,000测试图像的手写数字的数据集，图像的维度是28x28。下图是取自测试集中的样例：

![MNIST sample images](images\mnist_examples.png)
