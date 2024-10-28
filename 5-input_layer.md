
# Keras源码分析(5)：Tensor、Input 和 InputLayer

文件：/keras/engine/input_layer.py

Tensor，翻译过来就是张量。

TensorFlow支持两种类型的Tensor，一种是立即执行的（eager execution），一种是图执行的（graph execution）。图执行的Tensor一般只是一个空壳（placeholder），没有值（tensor_content），它只定义了维度（shape）和数据类型（dtype），当然还有部分定义好的方法，待未来填值计算。TensorFlow将那些待计算的tensor组织成有向图(graph)，后面tensor的计算依赖它前面的tensor的计算结果，因此，只要位于图入口处的那些tensor填入了值，就可执行图计算，就能输出结果。

Keras是建立在Tensorflow，CNTK或Theano之上的是一个高级神经网络库。Keras中的Tensor对底层TensorFlow或Theano的张量进行了扩展，加入了如下两个属性：
_Keras_history: 保存最近作用于这个tensor上的Layer对象及有关元数据。
_keras_shape: 保存(batch_size, input_dim,)与输入数据有关的维度大小的元组

张量是深度学习框架中的一个非常核心的概念，模型（内部）的计算都是基于张量进行的。

例子：
```sh
from keras.layers import Input, Dense
from keras.models import Model

inputs = Input(shape=(784,))

x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels)
```
上述的inputs和predictions都是tensor。显然，这里的tensor是一个图执行的tensor，由model.fit填入数据，然后执行计算。

下面来看Input是如何定义的：
```sh
def Input(shape=None, batch_shape=None,
          name=None, dtype=None, sparse=False,
          tensor=None):
    ......
    if shape is not None and not batch_shape:
        batch_shape = (None,) + tuple(shape)
    if not dtype:
        dtype = K.floatx()
    input_layer = InputLayer(batch_input_shape=batch_shape,
                             name=name, dtype=dtype,
                             sparse=sparse,
                             input_tensor=tensor)
    outputs = input_layer._inbound_nodes[0].output_tensors
    return unpack_singleton(outputs)
```
代码中引入了一个InputLayer的对象，它是整个网络的起始输入层，它实际做的只是对原始数据的原样输出。从这里可知这个Input实际上在内部定义了一个输入层，并把该输入层的输出作为自己的输出返回， 也即返回这里的input_tensor，如果它不是None;否则返回下面代码中的K.placeholder对象。
unpack_singleton(outputs)：如果outputs只有一个元素，则返回outputs[0]，否则返回outputs。

下面我们来看InputLayer的源码：
```sh
class InputLayer(Layer):

    @interfaces.legacy_input_support
    def __init__(self, input_shape=None, batch_size=None,
                 batch_input_shape=None,
                 dtype=None, input_tensor=None, sparse=False, name=None):
        ......
        self.trainable = False
        self.built = True
        self.sparse = sparse
        self.supports_masking = True
        ......
        self.batch_input_shape = batch_input_shape
        self.dtype = dtype

        if input_tensor is None:
            self.is_placeholder = True
            input_tensor = K.placeholder(shape=batch_input_shape,
                                         dtype=dtype,
                                         sparse=self.sparse,
                                         name=self.name)
        else:
            self.is_placeholder = False
            input_tensor._keras_shape = batch_input_shape

        input_tensor._uses_learning_phase = False
        input_tensor._keras_history = (self, 0, 0)
        Node(self,
             inbound_layers=[],
             node_indices=[],
             tensor_indices=[],
             input_tensors=[input_tensor],
             output_tensors=[input_tensor],
             input_masks=[None],
             output_masks=[None],
             input_shapes=[batch_input_shape],
             output_shapes=[batch_input_shape])
```
其中省略了一些对函数参数处理的代码。 对于NN中的每一层，不外乎三个部分：输入、处理（data的正向传播和error的反向传播）和输出。 在InputLayer的这段init的代码中，首先是对输入input_tensor的处理，如果是None， 则产生一个placeholder作为input tensor。从这里可看到所谓的tensor的占位符的特性。
接下来是一个逻辑处理和理解上的一个关键部分：new一个输入节点（Node）对象。Node对象的作用是用来联结两个层，我们将另辟章节对它进行分析，在这里我们只粗略地讨论一下传给它的参数。
Node的参数有4个主要部分：

（1）layers，包括outbound_layer和inbound_layers，这里outbound_layer接受的是这个InputLayer对象本身（self)，inbound_layers=[],因为InputLayer是第一个输入层，所以它的inbound_layers是空。

（2）tensors，包括输入张量和输出张量，它们都等于[input_tensor]，即：input_tensors=[input_tensor],output_tensors=[input_tensor]，所以说这一层其实什么也没干。

（3）shapes，很自然地，输入和输出的维度参数（包括batch_size）都是[batch_input_shape]，即：input_shapes=[batch_input_shape], output_shapes=[batch_input_shape]

（4）indices，这是一个很难理解的东东，需要在后面展开来讨论。 这里且注意这样一个细节就是output tensor(这里是input_tensor）的_keras_history，它的值的形式是（layer, node_index, tensor_index）这样的3元组，这里赋的值是(self, 0, 0)。
