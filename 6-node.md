# Keras源码分析(6)：Node

文件：/keras/engine/base_layer.py

在上一节，我们提到了Node，它的作用是用来联结两个层。这一节我们具体来看一看Node是如何联结两个层的。

对于Node这个命名，一开始会让人觉提很纳闷：为什么叫Node，难道它是一个神经元节点吗？它到底是什么，我们还是来看看源码吧。
```sh
class Node(object):
    def __init__(self, outbound_layer,
                 inbound_layers, node_indices, tensor_indices,
                 input_tensors, output_tensors,
                 input_masks, output_masks,
                 input_shapes, output_shapes,
                 arguments=None):
```
从Node对象的成员变量，我们来看看Node对象中到底保存了什么:
（1）首先是输出层，也是该Node对象被new出来的那个层，该层可以看作是我们分析Node的立足点或参照点
```sh
        self.outbound_layer = outbound_layer
```
（2）输入层列表，因为可能有多个输入，所以是个列表
```
        self.inbound_layers = inbound_layers
```
（3）输入和输出张量列表，因为可能有多个输入输出，且它们一一对应
```
        self.input_tensors = input_tensors
        self.output_tensors = output_tensors
```
（4）输入和输出mask张量列表，它们也是和输入输出一一对应
```
        self.input_masks = input_masks
        self.output_masks = output_masks
```
（5）输入和输出张量shape列表，它们也是和输入输出一一对应
```
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
```
（6）要解释清楚下面的变量，需要多一些口舌，解释我们放在后面。
```
        self.node_indices = node_indices
        self.tensor_indices = tensor_indices
```
这就是说，Node对象要保存的信息都是从外部传进来的，所以要想真正全面理解Node，我们还需要深入探寻创建和使用它的代码的上下文。

下面是Node对象的关键处理：它把自己分别追加到outbound_layer._inbound_nodes和inbound_layers中的每个layer._outbound_nodes的列表中。这样做的目的，显然是建立起从输入层到输出层关联。现在你可以画一个流程图，左边是几个输入层，右边是一个输出层，中间再画一个圆圈并标上Node，用线把它与左右两边的层连起来，此时，你看这个Node像节点吗？
```
        for layer in inbound_layers:
            if layer is not None:
                layer._outbound_nodes.append(self)
        outbound_layer._inbound_nodes.append(self)
```

现在我们回头来解释上面的第（6）条：

首先说说Node对象是在什么时候产生的。每当我们要建立一个网络时，我们需要new一些层，当传入tensors给它们的时候，就会调用这些层的__call__方法，这些层的__call__方法每调用一次，就会产生一个新的Node对象，并把当前的layer作为outbound_layer传给该Node对象，接下来该Node对象把自己分别放到outbound_layer._inbound_nodes列表和每一个inbound_layers._outbound_nodes的列表中，这一点我们在上面已经看到了。对于单输入输出，这种映射关系很清楚，到此一切都OK，但是对于有多个输入源的层，则问题就复杂了点。例如：
```
   a = Input(shape=(280, 256))
   b = Input(shape=(280, 256))
   lstm = LSTM(32)
   encoded_a = lstm(a)
   encoded_b = lstm(b)
```
当第一次调用lstm(a)的时候，它创建了一个node，并把这个node放到lstm的_inbound_nodes的第0个位置，第二次调用lstm(b)的时候，它又创建了一个node，并把这个node放到lstm的_inbound_nodes的第1个位置。当我们要想获取它们的输出时，我们可以这样做：
    encoded_a = lstm.get_output_at(0) 
    encoded_a = lstm.get_output_at(1)
这就是说，我们想要得到lstm的某个输出，不是简单地用lstm.output，而是用lstm.get_output_at(index)。

像上面这种情况，一个layer的输入可能是多个tensor的列表，进而产生的输出也是多个输出tensor的列表，所以，由这样的层作为inbound_layer的时候，我们需要知道input_tensor在inbound_layer的输出tensor列表中的位置，即：tensor_index。

对于像这样一个layer对象的输入和输出可能是一个列表的情况，我们要想从output_tensors中得到某个输出，则必须指定它在这个输出列表中的索引（tensor_index）。

从例子中我们把思路拉回来，考虑当前layer（outbound_layer）的所有输入，把与它们相关的node_index和tensor_index收集起来，分别依次放进两个node_indices和tensor_indices中，然后传递给该新建的node，并保存在它相应的成员变量中。这样，结合tensor._keras_history，我们把输出和输入的关系也就建立起来了。

尽管只是简单几行代码，解释起来并不简单。总之，通过许多Node对象把层和层，输入和输出连成了一个有向图，让它们彼此可以相互追溯。现在你可以再画一个图，左边是输入，右边是输出，中间再画一个圆圈并标上Node，用线把它与左右两边连起来，现在你再看这个Node像节点吗？