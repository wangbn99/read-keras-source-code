# Keras源码分析(10)：Model

文件：/keras/engine/training.py

把Model类放在training.py文件中，说明它肯定与训练有关，由下面Model的定义我们知道，它继承自Network，所以，Model具有Network和训练的功能，此外它还具有什么功能呢？
在Model类主要有四大功能模块： compile, fit, evaluate和predict。下面，我们来一步步解析。
```
class Model(Network):
```

## 一、compile：模型编译。用于配置训练模型。用compile接受的每一个参数对model进行配置。
```
def compile(self, optimizer,
                loss=None,
                metrics=None,
                loss_weights=None,
                sample_weight_mode=None,
                weighted_metrics=None,
                target_tensors=None,
                **kwargs):

        self.optimizer = optimizers.get(optimizer)
        self.loss = loss or []
        self.metrics = metrics or []
        self.loss_weights = loss_weights
        self.sample_weight_mode = sample_weight_mode
        self.weighted_metrics = weighted_metrics
```
处理参数loss，准备损失函数（loss functions）。有2种情形：

（1） 传入的loss是个损失函数的字典或列表，它对应着模型的多个输出，在每个输出上使用不同的损失；

（2） loss只是一个损失函数的名称，如果模型有多个输出，则所有的输出都使用相同的损失函数。

不管是哪种，模型最小化的损失值将是所有单个损失的总和。
```
if isinstance(loss, dict):
            loss_functions = []
            for name in self.output_names:
                loss_functions.append(losses.get(loss.get(name)))
        elif isinstance(loss, list):
            loss_functions = [losses.get(l) for l in loss]
        else:
            loss_function = losses.get(loss)
            loss_functions = [loss_function for _ in range(len(self.outputs))]
        self.loss_functions = loss_functions

        weighted_losses = [
            weighted_masked_objective(fn) for fn in loss_functions]
        skip_target_indices = []
        skip_target_weighing_indices = []
        self._feed_outputs = []
        self._feed_output_names = []
        self._feed_output_shapes = []
        self._feed_loss_fns = []
        for i in range(len(weighted_losses)):
            if weighted_losses[i] is None:
                skip_target_indices.append(i)
                skip_target_weighing_indices.append(i)
```
处理损失权重loss_weights，它是用以衡量损失函数对不同的模型输出的贡献。 模型将最小化的误差值是由 loss_weights 对每个输出上的损失进行加权的加权总和误差。
```
if loss_weights is None:
            loss_weights_list = [1. for _ in range(len(self.outputs))]
        elif isinstance(loss_weights, dict):
            loss_weights_list = []
            for name in self.output_names:
                loss_weights_list.append(loss_weights.get(name, 1.))
        elif isinstance(loss_weights, list):
            loss_weights_list = loss_weights
        else:
            raise TypeError('Could not interpret loss_weights argument: ')
```
处理target_tensors，创建模型的目标（targets of model）。
如果传入的参数target_tensors不为None,即下面的code，说明要使用外部指定的目标张量，它可以是单个张量（单输出模型），张量列表，或一个映射输出名称到目标张量的字典。
```
self.targets = []
        self._feed_targets = []
        if target_tensors is not None:
            if isinstance(target_tensors, list):
            elif isinstance(target_tensors, dict):
                tmp_target_tensors = []
                for name in self.output_names:
                    tmp_target_tensors.append(target_tensors.get(name, None))
                target_tensors = tmp_target_tensors
            elif K.is_tensor(target_tensors):
                target_tensors = [target_tensors]
            else:
                raise TypeError('Expected `target_tensors` to be a tensor')
```
如果target_tensors为None(默认情况),更或者是其中的某个为None,Keras 将为模型的目标创建一个占位符，在训练过程中将使用目标数据。
```
for i in range(len(self.outputs)):
            if i in skip_target_indices:
                self.targets.append(None)
            else:
                shape = K.int_shape(self.outputs[i])
                name = self.output_names[i]
                if target_tensors is not None:
                    target = target_tensors[i]
                else:
                    target = None
                if target is None or K.is_placeholder(target):
                    if target is None:
                        target = K.placeholder(
                            ndim=len(shape),
                            name=name + '_target',
                            sparse=K.is_sparse(self.outputs[i]),
                            dtype=K.dtype(self.outputs[i]))
                    self._feed_targets.append(target)
                    self._feed_outputs.append(self.outputs[i])
                    self._feed_output_names.append(name)
                    self._feed_output_shapes.append(shape)
                    self._feed_loss_fns.append(self.loss_functions[i])
                else:
                    skip_target_weighing_indices.append(i)
                self.targets.append(target)
```
处理样本权重模式sample_weight_mode，有两种情况：

（1） temporal： 即表示要执行按时间步采样权重（2D权重）；

（2） None，这是默认，为采样权重（1D）。

如果模型有多个输出，则可以传递一个 mode 字典或列表，以指示在每个输出上使用指定的
sample_weight_mode。
```
sample_weights = []
        sample_weight_modes = []
        if isinstance(sample_weight_mode, dict):
            for i, name in enumerate(self.output_names):
                if i in skip_target_weighing_indices:
                    weight = None
                    sample_weight_modes.append(None)
                else:
                    if sample_weight_mode.get(name) == 'temporal':
                        weight = K.placeholder(ndim=2,
                                               name=name + '_sample_weights')
                        sample_weight_modes.append('temporal')
                    else:
                        weight = K.placeholder(ndim=1,
                                               name=name + '_sample_weights')
                        sample_weight_modes.append(None)
                sample_weights.append(weight)
        elif isinstance(sample_weight_mode, list):
            for i in range(len(self.output_names)):
                if i in skip_target_weighing_indices:
                    weight = None
                    sample_weight_modes.append(None)
                else:
                    mode = sample_weight_mode[i]
                    name = self.output_names[i]
                    if mode == 'temporal':
                        weight = K.placeholder(ndim=2,
                                               name=name + '_sample_weights')
                        sample_weight_modes.append('temporal')
                    else:
                        weight = K.placeholder(ndim=1,
                                               name=name + '_sample_weights')
                        sample_weight_modes.append(None)
                sample_weights.append(weight)
        else:
            for i, name in enumerate(self.output_names):
                if i in skip_target_weighing_indices:
                    sample_weight_modes.append(None)
                    sample_weights.append(None)
                else:
                    if sample_weight_mode == 'temporal':
                        sample_weights.append(
                            K.placeholder(ndim=2,
                                          name=name + '_sample_weights'))
                        sample_weight_modes.append('temporal')
                    else:
                        sample_weights.append(
                            K.placeholder(ndim=1,
                                          name=name + '_sample_weights'))
                        sample_weight_modes.append(None)
        self.sample_weight_modes = sample_weight_modes
        self._feed_sample_weight_modes = []
        for i in range(len(self.outputs)):
            if i not in skip_target_weighing_indices:
                self._feed_sample_weight_modes.append(
                    self.sample_weight_modes[i])

        self.metrics_names = ['loss']
        self.metrics_tensors = []
```
计算总损失（total loss）： dot(output_loss, loss_weight) + self.losses
```
total_loss = None
        with K.name_scope('loss'):
            for i in range(len(self.outputs)):
                if i in skip_target_indices:
                    continue
                y_true = self.targets[i]
                y_pred = self.outputs[i]
                weighted_loss = weighted_losses[i]
                sample_weight = sample_weights[i]
                mask = masks[i]
                loss_weight = loss_weights_list[i]
                with K.name_scope(self.output_names[i] + '_loss'):
                    output_loss = weighted_loss(y_true, y_pred,
                                                sample_weight, mask)
                if len(self.outputs) > 1:
                    self.metrics_tensors.append(output_loss)
                    self.metrics_names.append(self.output_names[i] + '_loss')
                if total_loss is None:
                    total_loss = loss_weight * output_loss
                else:
                    total_loss += loss_weight * output_loss
            if total_loss is None:
                if not self.losses:
                    raise ValueError('The model cannot be compiled '
                                     'because it has no loss to optimize.')
                else:
                    total_loss = 0.

            for loss_tensor in self.losses:
                total_loss += loss_tensor
```
处理metrics，metrics指定了训练和测试期间的模型评估指标。可以为多输出模型的不同输出指定不同的评估指标，它可以是一个dict字典或list列表，如 metrics = {'output_a'：'accuracy'}。通常指标名称可以用全名，如：accuracy，crossentropy等，也可能简写，如：acc,ce等。
```
nested_metrics = collect_metrics(metrics, self.output_names)
        nested_weighted_metrics = collect_metrics(weighted_metrics,
                                                  self.output_names)
        self.metrics_updates = []
        self.stateful_metric_names = []
        self.stateful_metric_functions = []

        def handle_metrics(metrics, weights=None):
            metric_name_prefix = 'weighted_' if weights is not None else ''

            for metric in metrics:
                if metric in ('accuracy', 'acc', 'crossentropy', 'ce'):
                    output_shape = K.int_shape(self.outputs[i])
                    if (output_shape[-1] == 1 or
                       self.loss_functions[i] == losses.binary_crossentropy):
                        if metric in ('accuracy', 'acc'):
                            metric_fn = metrics_module.binary_accuracy
                        elif metric in ('crossentropy', 'ce'):
                            metric_fn = metrics_module.binary_crossentropy
                    elif (self.loss_functions[i] ==
                          losses.sparse_categorical_crossentropy):
                        if metric in ('accuracy', 'acc'):
                            metric_fn = metrics_module.sparse_categorical_accuracy
                        elif metric in ('crossentropy', 'ce'):
                            metric_fn = (
                                metrics_module.sparse_categorical_crossentropy)
                    else:
                        if metric in ('accuracy', 'acc'):
                            metric_fn = metrics_module.categorical_accuracy
                        elif metric in ('crossentropy', 'ce'):
                            metric_fn = metrics_module.categorical_crossentropy
                    if metric in ('accuracy', 'acc'):
                            suffix = 'acc'
                    elif metric in ('crossentropy', 'ce'):
                            suffix = 'ce'
                    weighted_metric_fn = weighted_masked_objective(metric_fn)
                    metric_name = metric_name_prefix + suffix
                else:
                    metric_fn = metrics_module.get(metric)
                    weighted_metric_fn = weighted_masked_objective(metric_fn)
                    if hasattr(metric_fn, 'name'):
                        metric_name = metric_fn.name
                    else:
                        metric_name = metric_fn.__name__
                    metric_name = metric_name_prefix + metric_name

                with K.name_scope(metric_name):
                    metric_result = weighted_metric_fn(y_true, y_pred,
                                                       weights=weights,
                                                       mask=masks[i])

                if len(self.output_names) > 1:
                    metric_name = self.output_names[i] + '_' + metric_name
                j = 1
                base_metric_name = metric_name
                while metric_name in self.metrics_names:
                    metric_name = base_metric_name + '_' + str(j)
                    j += 1
                self.metrics_names.append(metric_name)
                self.metrics_tensors.append(metric_result)

                if isinstance(metric_fn, Layer) and metric_fn.stateful:
                    self.stateful_metric_names.append(metric_name)
                    self.stateful_metric_functions.append(metric_fn)
                    self.metrics_updates += metric_fn.updates
        with K.name_scope('metrics'):
            for i in range(len(self.outputs)):
                if i in skip_target_indices:
                    continue

                y_true = self.targets[i]
                y_pred = self.outputs[i]
                weights = sample_weights[i]
                output_metrics = nested_metrics[i]
                output_weighted_metrics = nested_weighted_metrics[i]
                handle_metrics(output_metrics)
                handle_metrics(output_weighted_metrics, weights=weights)
```

为梯度和状态更新做准备

```
        self.total_loss = total_loss
        self.sample_weights = sample_weights
        self._feed_sample_weights = []
        for i in range(len(self.sample_weights)):
            if i not in skip_target_weighing_indices:
                self._feed_sample_weights.append(sample_weights[i])
```

为了节省时间，对于训练函数、测试函数和预测函数设置的惰性编译

```
        self._function_kwargs = kwargs
        self.train_function = None
        self.test_function = None
        self.predict_function = None

        trainable_weights = self.trainable_weights
        self._collected_trainable_weights = trainable_weights
```

## 二、fit：模型训练。在所有的fit参数中，x为训练数据，y为标签数据，validation_split指定有多少比例的训练数据用作验证数据，validation_data为验证数据集，epochs为训练轮次，batch_size为批大小。
```
def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            **kwargs):
```

对用户输入的数据进行校验，并转换成适合模型处理的标准数据格式
```
        x, y, sample_weights = self._standardize_user_data(
            x, y,
            sample_weight=sample_weight,
            class_weight=class_weight,
            batch_size=batch_size)
```

处理验证数据：有两种情况：

（1）是否需要验证：通过置do_validation决定，缺省是False，即不需要；但如果传入了参数validation_data或者validation_split或者validation_steps，则do_validation=True，意味着需要验证；

（2）验证数据产生，与下面if分支相对应：

a)由参数validation_data直接传入；否则

b)由validation_split指定一个划分比例，从训练数据中分出一部分作为验证数据；否则

c)当指定了validation_steps，一般与steps_per_epoch结合使用，这里validation_data则为测试数据和验证数据的生成器，本参数指定验证数据生成器的返回次数。

验证函数的输入是这种形式的元组：(val_x, val_y, val_sample_weights)或者(val_x, val_y, val_sample_weights, lr)，其中，val_x： 验证数据, val_y： 验证数据标签, val_sample_weights： 样本权重, lr： 学习速率。
```
do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data
            else:
                raise ValueError('When passing validation_data, '
                                 'it must contain 2 (x_val, y_val) '
                                 'or 3 (x_val, y_val, val_sample_weights) '
                                 'items, however it contains %d items' %
                                 len(validation_data))

            val_x, val_y, val_sample_weights = self._standardize_user_data(
                val_x, val_y,
                sample_weight=val_sample_weight,
                batch_size=batch_size)
            if self._uses_dynamic_learning_phase():
                val_inputs = val_x + val_y + val_sample_weights + [0.]
            else:
                val_inputs = val_x + val_y + val_sample_weights

        elif validation_split and 0. < validation_split < 1.:
            if any(K.is_tensor(t) for t in x):
                raise ValueError(
                    'If your data is in the form of symbolic tensors, '
                    'you cannot use `validation_split`.')
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(int(x[0].shape[0]) * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))
            sample_weights, val_sample_weights = (
                slice_arrays(sample_weights, 0, split_at),
                slice_arrays(sample_weights, split_at))
            if self._uses_dynamic_learning_phase():
                val_inputs = val_x + val_y + val_sample_weights + [0.]
            else:
                val_inputs = val_x + val_y + val_sample_weights

        elif validation_steps:
            do_validation = True
            if self._uses_dynamic_learning_phase():
                val_inputs = [0.]
```
为训练准备输入数组和训练函数。训练函数的输入是这种形式的元组：(x, y, sample_weights) 或者 (x, y, sample_weights, lr)，其中，x： 训练数据, y： 标签, sample_weights： 样本权重, lr： 学习速率。
```
if self._uses_dynamic_learning_phase():
            fit_inputs = x + y + sample_weights + [1.]
        else:
            fit_inputs = x + y + sample_weights
        self._make_train_function()
        fit_function = self.train_function

        out_labels = self.metrics_names
```

准备输验证函数:
```
        if do_validation:
            self._make_test_function()
            val_function = self.test_function
            callback_metrics = copy.copy(out_labels) + [
                'val_' + n for n in out_labels]
        else:
            callback_metrics = copy.copy(out_labels)
            val_function = None
            val_inputs = []
```

由training_arrays.fit_loop实现循环训练逻辑:
```
        return training_arrays.fit_loop(self, fit_function, fit_inputs,
                                        out_labels=out_labels,
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        verbose=verbose,
                                        callbacks=callbacks,
                                        val_function=val_function,
                                        val_inputs=val_inputs,
                                        shuffle=shuffle,
                                        callback_metrics=callback_metrics,
                                        initial_epoch=initial_epoch,
                                        steps_per_epoch=steps_per_epoch,
                                        validation_steps=validation_steps)
```

## 三、evaluate： 模型评估。在测试模式下对模型进行评估，按batch计算模型的误差损失值和其它可能的评估指标量。其代码逻辑与fit类似。
```
def evaluate(self, x=None, y=None,
                 batch_size=None,
                 verbose=1,
                 sample_weight=None,
                 steps=None):
```

对用户输入的数据进行校验，并转换成适合模型处理的标准数据格式
```
        x, y, sample_weights = self._standardize_user_data(
            x, y,
            sample_weight=sample_weight,
            batch_size=batch_size)
```
为评估准备输入数组和测试函数
```
        if self._uses_dynamic_learning_phase():
            ins = x + y + sample_weights + [0.]
        else:
            ins = x + y + sample_weights
        self._make_test_function()
        f = self.test_function
```
由training_arrays.test_loop实现循环评估逻辑:
```
        return training_arrays.test_loop(self, f, ins,
                                         batch_size=batch_size,
                                         verbose=verbose,
                                         steps=steps)
```

### 四、predict：预测。对输入的数据x进行预测，输出为对应的预测值（numpy array）
```
def predict(self, x,
                batch_size=None,
                verbose=0,
                steps=None):
```

对用户输入的数据进行校验，并转换成适合模型处理的标准数据格式
```
        x, _, _ = self._standardize_user_data(x)
        if self.stateful:
            if x[0].shape[0] > batch_size and x[0].shape[0] % batch_size != 0:
                raise ValueError('In a stateful network, '
                                 'you should only pass inputs with '
                                 'a number of samples that can be '
                                 'divided by the batch size. Found: ' +
                                 str(x[0].shape[0]) + ' samples. '
                                 'Batch size: ' + str(batch_size) + '.')
```

为预测准备输入数组和预测函数
```
        if self._uses_dynamic_learning_phase():
            ins = x + [0.]
        else:
            ins = x
        self._make_predict_function()
```

由training_arrays.predict_loop实现预测逻辑:
```
        f = self.predict_function
        return training_arrays.predict_loop(self, f, ins,
                                            batch_size=batch_size,
                                            verbose=verbose,
                                            steps=steps)
```
 