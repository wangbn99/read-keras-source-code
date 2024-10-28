# Keras源码分析(7)：Layer

文件：/keras/engine/base_layer.py

在前面第5节我们已经接触到一个简单而特别的层：InputLayer，它继承自基类：Layer。尽管简单，但关于层的基本逻辑倒也清楚明白。

在Node一节中，我们也说了许多与Node相关的Layer的内容。接下来我们仍需对这个基类Layer作进一步讨论。
```
class Layer(object):
```

这是Layer的主要处理逻辑所在，给定输入，产生输出。因为各种layer作用的不同，所以在些基类中无法做具体实现，它们的具体功能留待各种子类去实现。
```
    def call(self, inputs, **kwargs):
        return inputs
```

这个函数对call进行了进一步的包装。下面的分析中,我们仅保留了代码的主要部分，而省略了一些次要的代码，目的是让它的逻辑显得更加清晰。
```
    def __call__(self, inputs, **kwargs):
```

根据self.built标志确定是否需要build。如果需要，则首先要从inputs中获取input_shapes，这是build函数所唯一需要的，然后，把收集到的input_shapes传给build，执行build操作。build的作用在下面有解释，在此基类中build实际是个空操作。
```
        if not self.built:
            input_shapes = []
            for x_elem in to_list(inputs):
                if hasattr(x_elem, '_keras_shape'):
                    input_shapes.append(x_elem._keras_shape)
                elif hasattr(K, 'int_shape'):
                    input_shapes.append(K.int_shape(x_elem))
                else:
                    raise ValueError('......')

            self.build(unpack_singleton(input_shapes))
```

此处调用call，实现本层的主要逻辑，获得output。在此基类中call函数仅仅是将inputs原样奉还。
```
        output = self.call(inputs, **kwargs)
        ......

        # 如果call未对inputs进行修改，为避免inputs元数据的丢失，需要对它进行复制
        output_ls = to_list(output)
        inputs_ls = to_list(inputs)
        output_ls_copy = []
        for x in output_ls:
            if x in inputs_ls:
                x = K.identity(x)
            output_ls_copy.append(x)
        output = unpack_singleton(output_ls_copy)
        ......
 
        # 调用_add_inbound_node创建层间连接并保存历史
        self._add_inbound_node(input_tensors=inputs,
                               output_tensors=output,
                               input_masks=previous_mask,
                               output_masks=output_mask,
                               input_shapes=input_shape,
                               output_shapes=output_shape,
                               arguments=user_kwargs)
        ......

        #返回本层输出
        return output
```


这是一个内部方法，用来创建输入方向的node
```
    def _add_inbound_node(self, input_tensors, output_tensors,
                          input_masks, output_masks,
                          input_shapes, output_shapes, arguments=None):
        input_tensors = to_list(input_tensors)
        output_tensors = to_list(output_tensors)
        input_masks = to_list(input_masks)
        output_masks = to_list(output_masks)
        input_shapes = to_list(input_shapes)
        output_shapes = to_list(output_shapes)
```

遍历input_tensors，从每个input_tensor._keras_history中得到input_layer，node_index和tensor_index，把它们分别放进inbound_layers，node_indices和tensor_indices中。
```
        inbound_layers = []
        node_indices = []
        tensor_indices = []
        for x in input_tensors:
            if hasattr(x, '_keras_history'):
                inbound_layer, node_index, tensor_index = x._keras_history
                inbound_layers.append(inbound_layer)
                node_indices.append(node_index)
                tensor_indices.append(tensor_index)
            else:
                inbound_layers.append(None)
                node_indices.append(None)
                tensor_indices.append(None)
```

从前面Node一节我们已经知道，这个新的Node对象将把自己加到当前layer的_inbound_nodes列表中，同时也加到所有inbound_layer的_outbound_nodes列表中。
```
        Node(
            self,
            inbound_layers=inbound_layers,
            node_indices=node_indices,
            tensor_indices=tensor_indices,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            input_masks=input_masks,
            output_masks=output_masks,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            arguments=arguments
        )
```

更新输出output_tensors的history, _keras_shape和_uses_learning_phase。
```
        for i in range(len(output_tensors)):
            output_tensors[i]._keras_shape = output_shapes[i]
            uses_lp = any(
                [getattr(x, '_uses_learning_phase', False)
                 for x in input_tensors])
            uses_lp = getattr(self, 'uses_learning_phase', False) or uses_lp
            output_tensors[i]._uses_learning_phase = getattr(
                output_tensors[i], '_uses_learning_phase', False) or uses_lp
            output_tensors[i]._keras_history = (self, len(self._inbound_nodes) - 1, i)
```

对于上面的代码，我们把解释的重点放在最后一句上，因为它让我们清楚地知道，在output tensor的_keras_history中到底放进了什么。

（1）self：输出该第i个tensor的layer对象

（2）len(self._inbound_nodes) - 1：与第i个tensor相关的node对象在当前layer._inbound_nodes中的位置，即node_index

（3）i: 当然是第i个tensor在output_tensors中的位置，即：tensor_index


根据input_shape，创建该层的权重weights。不同的层可能有不同的创建方法，故留待子类去实现
```
    def build(self, input_shape):
        self.built = True
```

获取layer与给定node相关的输出tensor(s)
```
    def get_output_at(self, node_index):
        return self._get_node_attribute_at_index(node_index,
                                                 'output_tensors',
                                                 'output')
```

获取在layer._inbound_nodes[node_index]位置的node中由attr指定属性的值values
```
    def _get_node_attribute_at_index(self, node_index, attr, attr_name):
        ......
        values = getattr(self._inbound_nodes[node_index], attr)
        return unpack_singleton(values)
```
