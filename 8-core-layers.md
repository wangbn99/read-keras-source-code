# Keras源码分析(8)：Core Layers

文件：/keras/layers/core.py

Keras在它的keras.layers.core.py模块中，定义了一系列常用的层，包括Dense全连接层、Activation激活层、Flatten展平层等等，它们都继承自基类 Layer，下面将对它们一一阅读并解析。

## 一、Masking层

它的目的用来跳过某些时间步。这个层的设置和使用有些让人费解，所以是网上关于层发疑问最多的：其中如何设置mask？它是如何工作的？是最核心的两个问题，这正是我们要重点分析的。

初始化Masking对象。我们的目的是要屏蔽Input Tensor中的某些特殊的时间步，即是要对所有列都含有某个特定值的某些时间步进行屏蔽，如[1., 1., 1., 1., 1.]，但不是对任意一种模式的时间步都可以屏蔽的。我们要把这样的一个值传递给Masking对象的__init__方法，Masking对象的__init__方法中接收这个值的参数就是mask_value。

mask_value的缺省值是 0., 所以如果你什么参数值都没传的话，那么就是要屏蔽Input Tensor中那些全为 0 的行。此外，要想让这个mask起作用的话，还必须设置self.supports_masking这个标志为True。
```
class Masking(Layer):
    def __init__(self, mask_value=0., **kwargs):
        super(Masking, self).__init__(**kwargs)
        self.supports_masking = True
        self.mask_value = mask_value
```

下列call函数让我们知道masking是如何工作的。其中有个技巧，就是运用逻辑运算“与”和“或”。首先,用“不等”（K.not_equal)对Input Tensor中的每个特征值与mask_value进行比较；然后用“任意一个”（K.any）逐时间步进行检查，检查结果是：除了所有特征值都是与mask_value相等的时间步，它的结果值为False外，其它的时间步的结果值都是True；再就是用K.cast把逻辑的False和True它们转换为浮点值的0.或1.；最后，把inputs与该转换过后的结果相乘。经过这样一些列的操作后，把要屏蔽的行的值都变成了0.，从而在后续的运算中，屏蔽了该时间步的作用。
```
def call(self, inputs):
        boolean_mask = K.any(K.not_equal(inputs, self.mask_value),
                             axis=-1, keepdims=True)
        return inputs * K.cast(boolean_mask, K.dtype(inputs))

计算output shape，它与input shape一致。
    def compute_output_shape(self, input_shape):
        return input_shape
```

## 二、Dropout层

该层的作用是要在网络的训练阶段按一定的比率（rate）随机地（在Tensorflow中是用一致分布）丢弃（屏蔽）部分输入单元，目的是为了防止过配。
它的初始化参数中有两个是可想而知的：

（1）rate：取[0，1]之间的一个浮点娄，设置丢弃（屏蔽）输入单元的比例。

（2）seed: Python整数用作随机数种子。


而另外一个参数noise_shape，从字面看倒是很直接，但理解起来却不那么容易，它是用来指定要对哪些维做dropout。举个例子：设input shape是(batch_size, timesteps, features)，我们要对timesteps维做dropout，那么我们可能这样设置noise_shape：noise_shape=(batch_size, 1, features)。这就是说，我们把noise_shape与input_shape逐维进行比较，如果不同且noise_shape将该维置为1的，即是要对该维做dropout。

参考：Dropout: A Simple Way to Prevent Neural Networks from Overfitting
```
class Dropout(Layer):
    @interfaces.legacy_dropout_support
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(Dropout, self).__init__(**kwargs)
        self.rate = min(1., max(0., rate))
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = K.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)
```

实际dropout是由后台支持库做的，见K.dropout与K.in_train_phase。
```
    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape,
                                 seed=self.seed)
            return K.in_train_phase(dropped_inputs, inputs,
                                    training=training)
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape
```

## 三、SpatialDropout1D层

该层是Dropout层的运用于1D的特殊情况，对于这种case, 输入tensor的shape应该是(samples, timesteps, channels)，dropout将作用于timesteps维，所以noise_shape应该是(samples, 1, channels)。

参考：Efficient Object Localization Using Convolutional Networks
```
class SpatialDropout1D(Dropout):
```

输入张量的维度应该是3，所以传入参数ndim=3给InputSpec
```
    @interfaces.legacy_spatialdropout1d_support
    def __init__(self, rate, **kwargs):
        super(SpatialDropout1D, self).__init__(rate, **kwargs)
        self.input_spec = InputSpec(ndim=3)
```

计算noise_shape，timesteps维为1，其它维跟输入tensor相同
```
    def _get_noise_shape(self, inputs):
        input_shape = K.shape(inputs)
        noise_shape = (input_shape[0], 1, input_shape[2])
        return noise_shape
```

## 四、SpatialDropout2D层

该层是Dropout层的运用于2D图像的情况，对于这种case, 依据图像格式的不同，即channels在图像数据中位置的不同（channels_last或channels_first), 输入tensor的shape应该是(samples, rows, cols, channels)抑或是(samples, channels, rows, cols)，不管是哪一种，dropout将作用于rows维和cols维，所以相应地noise_shape应该是(samples, 1, 1, channels)抑或是(samples, channels, 1, 1)。

参考：Efficient Object Localization Using Convolutional Networks
```
class SpatialDropout2D(Dropout):
```
参数data_format用来指定图像格式，即channels_last或channels_first，如果参数没有指定的话，默认将取自你在 Keras 的配置文件 ~/.keras/keras.json 中设置的 image_data_format 的值，如果你从未设置过它，那么它将是 channels_last
```
@interfaces.legacy_spatialdropoutNd_support
    def __init__(self, rate, data_format=None, **kwargs):
        super(SpatialDropout2D, self).__init__(rate, **kwargs)
        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)

    def _get_noise_shape(self, inputs):
        input_shape = K.shape(inputs)
        if self.data_format == 'channels_first':
            noise_shape = (input_shape[0], input_shape[1], 1, 1)
        else:
            noise_shape = (input_shape[0], 1, 1, input_shape[3])
        return noise_shape
```


## 五、SpatialDropout3D层

理解了前面的1D和2D，SpatialDropout3Db也就很容易理解了。对于这种case, 依据图像格式的不同，即channels在图像数据中位置的不同（channels_last或channels_first), 输入tensor的shape应该是(samples, dim1, dim2, dim3, channels)抑或是(samples, channels, dim1, dim2, dim3)，不管是哪一种data_format，dropout将作用于dim1维、dim2维和dim3维，所以相应地noise_shape应该是(samples, 1, 1, 1, channels)抑或是(samples, channels, 1, 1, 1)。

参考：Efficient Object Localization Using Convolutional Networks
```
class SpatialDropout3D(Dropout):

    @interfaces.legacy_spatialdropoutNd_support
    def __init__(self, rate, data_format=None, **kwargs):
        super(SpatialDropout3D, self).__init__(rate, **kwargs)
        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=5)

    def _get_noise_shape(self, inputs):
        input_shape = K.shape(inputs)
        if self.data_format == 'channels_first':
            noise_shape = (input_shape[0], input_shape[1], 1, 1, 1)
        else:
            noise_shape = (input_shape[0], 1, 1, 1, input_shape[4])
        return noise_shape
```


## 六、Activation激活层

这个层非常简单，闵是将激活函数应用于输入。该激活函数由初始化参数 activation 指定。
```
class Activation(Layer):

    def __init__(self, activation, **kwargs):
        super(Activation, self).__init__(**kwargs)
        self.supports_masking = True
        self.activation = activations.get(activation)

    def call(self, inputs):
        return self.activation(inputs)

    def get_config(self):
        config = {'activation': activations.serialize(self.activation)}
        base_config = super(Activation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
```

## 七、Reshape层

对输入的尺寸重新调整。如果使用此层作为模型中的第一层，则需要使用参数 input_shape （其中不包括样本数samples的维）。在target_shape中的某一维可以使用 “-1”， 表示维度推断。
```
class Reshape(Layer):

    def __init__(self, target_shape, **kwargs):
        super(Reshape, self).__init__(**kwargs)
        self.target_shape = tuple(target_shape)
```

此内部函数将处理维度推断，变量unknown用来保存待推断维度的轴索引，变量known用来统计target_shape已知的维数,所以我们用所有input_shape中维数除以known, 就可推断待定的维数。
```
def _fix_unknown_dimension(self, input_shape, output_shape):

        output_shape = list(output_shape)
        msg = 'total size of new array must be unchanged'

        known, unknown = 1, None
        for index, dim in enumerate(output_shape):
            if dim < 0:
                if unknown is None:
                    unknown = index
                else:
                    raise ValueError('Can only specify one unknown dimension.')
            else:
                known *= dim

        original = np.prod(input_shape, dtype=int)
        if unknown is not None:
            if known == 0 or original % known != 0:
                raise ValueError(msg)
            output_shape[unknown] = original // known
        elif original != known:
            raise ValueError(msg)

        return tuple(output_shape)
```

计算输出shape
```
    def compute_output_shape(self, input_shape):
        if None in input_shape[1:]:
            # input shape (partially) unknown? replace -1's with None's
            return ((input_shape[0],) +
                    tuple(s if s != -1 else None for s in self.target_shape))
        else:
            # input shape known? then we can compute the output shape
            return (input_shape[0],) + self._fix_unknown_dimension(
                input_shape[1:], self.target_shape)
```

执行维度尺寸调整
```
    def call(self, inputs):
        return K.reshape(inputs, (K.shape(inputs)[0],) + self.target_shape)
```

## 八、Permute层

按照给定的模式重新排列输入维度的位置。它与前面讨论的Reshape层不同：
Reshape层改变输入数据的形状，但没改变输入数据的顺序，例如：

Reshape(dims=(2,-1))，作用在[[1, 2, 3, 4, 5, 6]]上，输出结果是：[[[ 1. 2. 3.], [ 4. 5. 6.]]]，

Reshape(dims=(3,-1)), 作用在[[1, 2, 3, 4, 5, 6]]上，输出结果是：[[[ 1. 2.],[ 3. 4.],[ 5. 6.]]]；

而Permute层则仅是转换了维度的位置，即顺序变了，但大小未变。例如：

Permute(dims=(2,1)), 作用在[[[1, 2, 3],[4, 5, 6]]]上，输出结果是：[[[ 1. 4.], [ 2. 5.], [ 3. 6.]]]。

请注意上述例子中输入数据的维度及顺序。对于Permute，它的输入数据的shape是（1，2，3）， 而经Permute置换后的output shape是（1，3，2），通过这个例子，我们可能很好地解释参数dims=(2,1)：它是一个模式，我们把它写成：dims[output_index-1]=input_index, 这里output_index从1开始，所以我们得到这样一个对应关系：output_index = 1，input_index = 2；output_index = 2，input_index = 1；因此，input shape是（1，2，3）,对应的output shape是（1，3，2），其中samples维保持不变。
```
class Permute(Layer):
```

因为dims中不包括samples维，所以ndim=len(dims) + 1
```
    def __init__(self, dims, **kwargs):
        super(Permute, self).__init__(**kwargs)
        self.dims = tuple(dims)
        self.input_spec = InputSpec(ndim=len(self.dims) + 1)
```
首先复制input_shape到output_shape，然后根据dims模式中指定的output_index和input_index，令output_shape[output_index] = input_shape[input_index]即可
```
def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        output_shape = copy.copy(input_shape)
        for i, dim in enumerate(self.dims):
            target_dim = input_shape[dim]
            output_shape[i + 1] = target_dim
        return tuple(output_shape)
```

实施置换
```
    def call(self, inputs):
        return K.permute_dimensions(inputs, (0,) + self.dims)
```

## 九、Flatten层

Flatten层用来将输入“展平”，即把多维输入一维化，常用于卷积层到全连接层的过渡。Flatten不影响批大小（batch size）。
```
class Flatten(Layer):
```

既然是要展开，当然输入的最小维度不低于3，即min_ndim=3
```
    def __init__(self, data_format=None, **kwargs):
        super(Flatten, self).__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=3)
        self.data_format = K.normalize_data_format(data_format)
```

展平后的维度大小计算，它应该是第0维batch大小不变，而第1维应该是input_shape中除了batch维之外的其它所有维大小的乘积
```
    def compute_output_shape(self, input_shape):
        if not all(input_shape[1:]):
            raise ValueError('The shape of the input to "Flatten" '
                             'is not fully defined '
                             '(got ' + str(input_shape[1:]) + '. '
                             'Make sure to pass a complete "input_shape" '
                             'or "batch_input_shape" argument to the first '
                             'layer in your model.')
        return (input_shape[0], np.prod(input_shape[1:]))
```

在展平之前，首先要老虑到channels的维的位置，对于channels_first情形，要用维度置换方法，变成channels_last，然后进行展平操作
```
    def call(self, inputs):
        if self.data_format == 'channels_first':
            # Ensure works for any dim
            permutation = [0]
            permutation.extend([i for i in
                                range(2, K.ndim(inputs))])
            permutation.append(1)
            inputs = K.permute_dimensions(inputs, permutation)

        return K.batch_flatten(inputs)
```

## 十、RepeatVector层

这一层很简单，它将2D输入重复指定的次数。如果输入shape是（batch_size, features，则输出将是（batch_size, n, features)。

```
class RepeatVector(Layer):
```

指定重复次数n
```
    def __init__(self, n, **kwargs):
        super(RepeatVector, self).__init__(**kwargs)
        self.n = n
        self.input_spec = InputSpec(ndim=2)
```

计算output shape
```
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n, input_shape[1])
```

重复操作
```
    def call(self, inputs):
        return K.repeat(inputs, self.n)
```


## 十一、Lambda层

Lambda层可将任意的表达式封装成 Layer 对象。该层非常适合于那种只想对流经该层的数据做变换，而不需要进行参数学习的情形。关于如何使用该层，Keras文档中有两个很好的例子可参阅。

```
class Lambda(Layer):
```

在该层的初始化参数中，function参数是不可或缺的，其定义形式象这样的：function(inputs, **arguments)，它接受输入tensor(s)作为它的第1个参数。
```
@interfaces.legacy_lambda_support
    def __init__(self, function, output_shape=None,
                 mask=None, arguments=None, **kwargs):
        super(Lambda, self).__init__(**kwargs)
        self.function = function
        self.arguments = arguments if arguments else {}
```
如果有mask,则须置supports_masking标志
```
        if mask is not None:
            self.supports_masking = True
        self.mask = mask
```

处理output_shape参数，有3种情形：

（1）可以是None
```
        if output_shape is None:
            self._output_shape = None
```
（2）可以是一个tuple或list
```
        elif isinstance(output_shape, (tuple, list)):
            self._output_shape = tuple(output_shape)
```
（3）可以是一个函数，
```
        else:
            if not callable(output_shape):
                raise TypeError('In Lambda, `output_shape` '
                                'must be a list, a tuple, or a function.')
            self._output_shape = output_shape
```

计算output shape，对应于output_shape的初始化3种情形
```
    def compute_output_shape(self, input_shape):
```
(1)由tensorflow或CNTK从inputs推断
```
        if self._output_shape is None:
            # With TensorFlow or CNTK, we can infer the output shape directly:
            if K.backend() in ('tensorflow', 'cntk'):
                if isinstance(input_shape, list):
                    xs = [K.placeholder(shape=shape) for shape in input_shape]
                    x = self.call(xs)
                else:
                    x = K.placeholder(shape=input_shape)
                    x = self.call(x)
                if isinstance(x, list):
                    return [K.int_shape(x_elem) for x_elem in x]
                else:
                    return K.int_shape(x)
            # Otherwise, we default to the input shape.
            warnings.warn('`output_shape` argument not specified for layer {} '
                          'and cannot be automatically inferred '
                          'with the Theano backend. '
                          'Defaulting to output shape `{}` '
                          '(same as input shape). '
                          'If the expected output shape is different, '
                          'specify it via the `output_shape` argument.'
                          .format(self.name, input_shape))
            return input_shape
```
（2）对于tuple或list，因为其中不包括样本大小维（或batch_size），所以须将input_shape中batch_size放丰output shape的最前
```
       elif isinstance(self._output_shape, (tuple, list)):
            if isinstance(input_shape, list):
                num_samples = input_shape[0][0]
            else:
                num_samples = input_shape[0] if input_shape else None
            return (num_samples,) + tuple(self._output_shape)
```
（3）将_output_shape函数作用于input_shape获取output shape
```
        else:
            shape = self._output_shape(input_shape)
            if not isinstance(shape, (list, tuple)):
                raise ValueError('`output_shape` function must return a tuple or '
                                 'a list of tuples.')
            if isinstance(shape, list):
                if isinstance(shape[0], int) or shape[0] is None:
                    shape = tuple(shape)
            return shape
```

进行函数调用
```
    def call(self, inputs, mask=None):
        arguments = self.arguments
        if has_arg(self.function, 'mask'):
            arguments['mask'] = mask
        return self.function(inputs, **arguments)
```


## 十二、Dense全连接层

该层实现以下操作：output = activation(dot(input, kernel) + bias)，这也是我们在经典的神经网络中非常熟知的在神经元节点中执行的计算公式。这里，activation是按元素计算的激活函数，kernel是由网络层创建的权值矩阵，bias是由网络层创建的偏置向量。
```
class Dense(Layer):
```

这里的这些初始化参数名称作为神经网络的名词，也是我们在经典神经网络中非常熟悉的，后面会有专门的解读。
```
    @interfaces.legacy_dense_support
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Dense, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True
```

根据input_shape，创建该层的权重kernel和bias
```
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
```
只在 use_bias 为 True 时才会创建bias
```
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True
```

实现操作：output = activation(dot(input, kernel) + bias)
```
    def call(self, inputs):
        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output
```

依据input_shape中的batch_size和参数units计算output shape
```
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)
```


## 十三、ActivityRegularization层

我们对英文文档的翻译就是：对基于代价函数的输入活动应用一个更新。这种解释让人一头雾水。我们的解释一点点深入：

（1）首先是初始化：它通过参数传入的两个正则因子l1和l2，初始化一个activity_regularizer=regularizers.L1L2(l1=l1, l2=l2)，它到底有什么用？

（2）接下来是我们的分析。一般情况下，我们对loss的计算包括两个部分：

a. 由数据本身计算得的loss,记作：DataLoss

b. 由数据的分部而增加的扰动，即正则项，记为: RegularizationLoss

因此，loss = DataLoss + RegularizationLoss

（3）RegularizationLoss双分为两种：

a. 基于权重的正则损失，对于这种情况，RegularizationLoss = f(Weights)

b. 基于输入活动的正则损失，对于这种情况，RegularizationLoss = f(outputs)

（4）在我们非常清楚输入数据集的分布的情况下，我们将应用基于输入活动的正则损失。就是这里的情形。
class ActivityRegularization(Layer):

```
    def __init__(self, l1=0., l2=0., **kwargs):
        super(ActivityRegularization, self).__init__(**kwargs)
        self.supports_masking = True
        self.l1 = l1
        self.l2 = l2
        self.activity_regularizer = regularizers.L1L2(l1=l1, l2=l2)

    def compute_output_shape(self, input_shape):
        return input_shape
```
