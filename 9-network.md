# Keras源码分析(9)：Network

文件：/keras/engine/network.py

在前面我们讨论Node的作用时说过，Node对象把层和层，输入和输出联结了起来，最终把一个多层网络连成了一个有向无环图（DAG），而Network对象正是实现这样的过程。
```
class Network(Layer):
```
首先，要说明的是，它为什么要继承自Layer，应该有这样几点：

（1）因为它要生成一个有向无环图（DAG），有入口也有出口，有输入也有输出，这一点跟Layer是一致的；

（2）它生成有向无环图（DAG）的目的是为了计算，为了触发DAG进行计算，就需要提供一个函数给外部调用，这个函数就是call, 这一点跟Layer也是一致的；

（3）模型作为一个Layer可嵌套使用。

DAG图的构建是在Network对象实例化时完成的。
```
    def __init__(self, *args, **kwargs):
        if (len(args) == 2 or
            len(args) == 1 and 'outputs' in kwargs or
                'inputs' in kwargs and 'outputs' in kwargs):
            self._init_graph_network(*args, **kwargs)
        else:
            self._init_subclassed_network(**kwargs)
```

由上可见，DAG图的构建实际上是由调用_init_graph_network完成的。下面我们略去了Network作为Layer的有关代码后，它的主要逻辑也看得很清楚了。它进一步调用位于同一文件中的外部_map_graph_network，从而获得了关键的几个数据结构nodes, nodes_by_depth, layers, layers_by_depth，并把它们保存到对象的相应的成员变量中。
```
def _init_graph_network(self, inputs, outputs, name=None):
        ......
        self.inputs = to_list(inputs, allow_tuple=True)
        self.outputs = to_list(outputs, allow_tuple=True)
        ......
        nodes, nodes_by_depth, layers, layers_by_depth = _map_graph_network(
            self.inputs, self.outputs)
        self._network_nodes = nodes
        self._nodes_by_depth = nodes_by_depth
        self._layers = layers
        self._layers_by_depth = layers_by_depth
        ......
```

现在，我们来看看_map_graph_network的代码：

```
def _map_graph_network(inputs, outputs):

    # 初始化几个数据结构用来保存与所要创建的图相关的nodes和layers信息
    network_nodes = set()  # ids of all nodes relevant to the Network
    nodes_depths = {}  # dict {node: depth value}
    layers_depths = {}  # dict {layer: depth value}
    layer_indices = {}  # dict {layer: index in traversal}
    nodes_in_decreasing_depth = []
```

下面build_map内部数据是一个递归函数，它是要从传入的tensor开始反向递归构建DAG图。
在递归构建过程中，那些相关的node将呈现下列3种状态中的一种：

（1）已完成（finished_nodes）

（2）进行中（nodes_in_progress）

（3）将要处理（layer._inbound_nodes[node_index]）

所有这些都作为参数传给了build_map函数。
```
    def build_map(tensor,
                  finished_nodes,
                  nodes_in_progress,
                  layer,
                  node_index,
                  tensor_index):

        # 获得将要处理的node
        node = layer._inbound_nodes[node_index]

        # 检查nodes_in_progress以避免环
        if node in nodes_in_progress:
            raise ValueError('The tensor ' + str(tensor) + ' at layer "' +
                             layer.name + '" is part of a cycle.')

        # 防止重复劳动
        if node in finished_nodes:
            return

        # 更新外层network_nodes集合
        node_key = _make_node_key(layer.name, node_index)
        network_nodes.add(node_key)

        # 更新外层layer_indices字典
        if layer not in layer_indices:
            layer_indices[layer] = len(layer_indices)

        # 开始处理该node
        nodes_in_progress.add(node)

        # 深度优先搜索那些连接到当前node的输入方向上的tensors
        for i in range(len(node.inbound_layers)):
            x = node.input_tensors[i]
            layer = node.inbound_layers[i]
            node_index = node.node_indices[i]
            tensor_index = node.tensor_indices[i]
            build_map(x, finished_nodes, nodes_in_progress, layer,
                      node_index, tensor_index)

        # 处理完毕，将当前node加到finished_nodes中，
        # 并从nodes_in_progress中移出
        finished_nodes.add(node)
        nodes_in_progress.remove(node)
        # 这是最为关键一步，它将所有nodes的遍历深度保存到
        # 队列nodes_in_decreasing_depth中，此队列将是后
        # 续正向计算和误差反向传播执行的依据
        nodes_in_decreasing_depth.append(node)
```

到此build_map结束，开始准备调用build_map函数。首先，初始化finished_nodes和nodes_in_progress，然后，从outputs循环开始调用build_map函数，按反向递归构建DAG图。
```
finished_nodes = set()
    nodes_in_progress = set()
    for x in outputs:
        layer, node_index, tensor_index = x._keras_history
        build_map(x, finished_nodes, nodes_in_progress,
                  layer=layer,
                  node_index=node_index,
                  tensor_index=tensor_index)
```
上述调用build_map函数产生了3个结果：

（1）network_nodes：所有nodes集合；

（2）layer_indices：layer->index字典；

（3）nodes_in_decreasing_depth：按遍历顺序保存的nodes列表。

其中结果（1）将被_map_graph_network返回，我们仍需用（2）和（3）来构建节点的深度、按深度递减的层列表和层的深度，即：nodes_by_depth, layers, layers_by_depth。
```
# 根据nodes_in_decreasing_depth计算
    # 节点到深度的映射nodes_depths：node->depth，
    # 以及层到深度的映射layers_depths：layer->depth
    for node in reversed(nodes_in_decreasing_depth):
        depth = nodes_depths.setdefault(node, 0)

        previous_depth = layers_depths.get(node.outbound_layer, 0)
        depth = max(depth, previous_depth)
        layers_depths[node.outbound_layer] = depth
        nodes_depths[node] = depth

        for i in range(len(node.inbound_layers)):
            inbound_layer = node.inbound_layers[i]
            node_index = node.node_indices[i]
            inbound_node = inbound_layer._inbound_nodes[node_index]
            previous_depth = nodes_depths.get(inbound_node, 0)
            nodes_depths[inbound_node] = max(depth + 1, previous_depth)

    # 按深度对节点进行分组，即：depth->nodes，从而得到nodes_by_depth
    nodes_by_depth = {}
    for node, depth in nodes_depths.items():
        if depth not in nodes_by_depth:
            nodes_by_depth[depth] = []
        nodes_by_depth[depth].append(node)

    # 按深度对层进行分组，即：depth->layers，从而得到layers_by_depth
    layers_by_depth = {}
    for layer, depth in layers_depths.items():
        if depth not in layers_by_depth:
            layers_by_depth[depth] = []
        layers_by_depth[depth].append(layer)

    # 按深度下降的顺序组织排列所有层
    depth_keys = list(layers_by_depth.keys())
    depth_keys.sort(reverse=True)
    layers = []
    for depth in depth_keys:
        layers_for_depth = layers_by_depth[depth]
        # 如果深度相同，则按遍历的顺序
        layers_for_depth.sort(key=lambda x: layer_indices[x])
        layers.extend(layers_for_depth)
    ......
```

    返回节点的集合、节点的深度、按深度递减的层列表和层的深度
```
    return network_nodes, nodes_by_depth, layers, layers_by_depth
```

Network继承自Layer，所以计算也是由Network对象的call方法完成。为了减少重复计算的开销，Network对象对同一inputs和masks的计算结果进行了缓存(self._output_tensor_cache)，如果已计算过了，则直接从缓存中取出；如果没有，则调用内部方法self.run_internal_graph进行计算。
```
def call(self, inputs, mask=None):
        inputs = to_list(inputs)
        if mask is None:
            masks = [None for _ in range(len(inputs))]
        else:
            masks = to_list(mask)
        cache_key = object_list_uid(inputs)
        cache_key += '_' + object_list_uid(masks)
        if cache_key in self._output_tensor_cache:
            return self._output_tensor_cache[cache_key]
        else:
            output_tensors, _, _ = self.run_internal_graph(inputs, masks)
            return output_tensors

    # 根据inputs计算network的输出tensors。
    def run_internal_graph(self, inputs, masks=None):
        if masks is None:
            masks = [None for _ in range(len(inputs))]
        
        # tensor_map中存放所有已计算过的tensors和masks
        # 用传入的inputs和masks初始化tensor_map
        tensor_map = {}
        for x, y, mask in zip(self.inputs, inputs, masks):
            tensor_map[str(id(x))] = (y, mask)

 # 依深度递减遍历
        depth_keys = list(self._nodes_by_depth.keys())
        depth_keys.sort(reverse=True)
        for depth in depth_keys:
            nodes = self._nodes_by_depth[depth]
            for node in nodes:
                layer = node.outbound_layer
                reference_input_tensors = node.input_tensors
                reference_output_tensors = node.output_tensors

                # 检查reference_input_tensors中的所有tensors是否都在tensor_map中
                # 即是否都已计算过了，把其中已计算过的放进computed_data
                computed_data = []  # List of tuples (input, mask).
                for x in reference_input_tensors:
                    if str(id(x)) in tensor_map:
                        computed_data.append(tensor_map[str(id(x))])
                # 检查的方法是比较computed_data和reference_input_tensors中元素的个数是否相等
                if len(computed_data) == len(reference_input_tensors):
                    # 相等则表示当前layer的所有input_tensors都已计算过了，所以可以调用layer.call方法了
                    with K.name_scope(layer.name):
                        if node.arguments:
                            kwargs = node.arguments
                        else:
                            kwargs = {}
                        if len(computed_data) == 1:
                            computed_tensor, computed_mask = computed_data[0]
                            if has_arg(layer.call, 'mask'):
                                if 'mask' not in kwargs:
                                    kwargs['mask'] = computed_mask
                            output_tensors = to_list(
                                layer.call(computed_tensor, **kwargs))
                            output_masks = layer.compute_mask(computed_tensor,
                                                              computed_mask)
                            if output_masks is None:
                                output_masks = [None for _ in output_tensors]
                            else:
                                output_masks = to_list(output_masks)
                            computed_tensors = [computed_tensor]

                            computed_masks = [computed_mask]
                        else:
                            computed_tensors = [x[0] for x in computed_data]
                            computed_masks = [x[1] for x in computed_data]
                            if has_arg(layer.call, 'mask'):
                                if 'mask' not in kwargs:
                                    kwargs['mask'] = computed_masks
                            output_tensors = to_list(
                                layer.call(computed_tensors, **kwargs))
                            output_masks = layer.compute_mask(computed_tensors,
                                                              computed_masks)
                            if output_masks is None:
                                output_masks = [None for _ in output_tensors]
                            else:
                                output_masks = to_list(output_masks)
                        if (hasattr(layer, 'activity_regularizer') and
                                layer.activity_regularizer is not None):
                            with K.name_scope('activity_regularizer'):
                                regularization_losses = [
                                    layer.activity_regularizer(x)
                                    for x in output_tensors]
                            layer.add_loss(regularization_losses,
                                           inputs=computed_tensors)

                    ......
                    # 把当前计算过的output_tensors和output_masks加入到tensor_map中
                    for x, y, mask in zip(reference_output_tensors,
                                          output_tensors,
                                          output_masks):
                        tensor_map[str(id(x))] = (y, mask)

        # 从tensor_map提取output_tensors和output_masks
        # 从output_tensors中提到output_shapes
        output_tensors = []
        output_masks = []
        output_shapes = []
        for x in self.outputs:
            assert str(id(x)) in tensor_map, 'Could not compute output ' + str(x)
            tensor, mask = tensor_map[str(id(x))]
            if hasattr(tensor, '_keras_shape') and output_shapes is not None:
                shape = tensor._keras_shape
                output_shapes.append(shape)
            else:
                output_shapes = None
            output_tensors.append(tensor)
            output_masks.append(mask)

        return output_tensors, output_masks, output_shapes
```
