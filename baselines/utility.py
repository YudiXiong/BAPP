from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
from collections import deque
import tensorflow as tf
import collections
import hashlib
import numbers

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.deprecation import deprecated

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


ASSERT_LIKE_RNNCELL_ERROR_REGEXP = "is not an RNNCell"

def to_pickled_df(data_directory, **kwargs):
    for name, df in kwargs.items():
        df.to_pickle(os.path.join(data_directory, name + '.df'))

def pad_history(itemlist,length,pad_item):
    if len(itemlist)>=length:
        return itemlist[-length:]
    if len(itemlist)<length:
        temp = [pad_item] * (length-len(itemlist))
        itemlist.extend(temp)
        return itemlist

def extract_axis_1(data, ind):
    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)

    return res

def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs

def calculate_hit_ndcg(sorted_list,topk,true_items,hit_purchase,ndcg_purchase):
    for i in range(len(topk)):


        rec_list = sorted_list[:, -topk[i]:]
        for j in range(len(true_items)):
            if true_items[j] in rec_list[j] :
                rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])[0][0]
                hit_purchase[i] += 1.0
                ndcg_purchase[i] += 1.0 / np.log2(rank + 1)


def calculate_mrr(sorted_list,topk,true_items,mrr_purchase):
    for i in range(len(topk)):
        rec_list = sorted_list[:, -topk[i]:]
        for j in range(len(true_items)):
            if true_items[j] in rec_list[j] :
                rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])[0][0]


                mrr_purchase[i] += 1.0 / (rank)


def calculate_hit(sorted_list,topk,true_items,rewards,r_click,total_reward,hit_click,ndcg_click,hit_purchase,ndcg_purchase):
    for i in range(len(topk)):
        rec_list = sorted_list[:, -topk[i]:]
        for j in range(len(true_items)):
            if true_items[j] in rec_list[j] :
                rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])[0][0]
                total_reward[i] += rewards[j]
                if rewards[j] == r_click:
                    hit_click[i] += 1.0
                    ndcg_click[i] += 1.0 / np.log2(rank + 1)
                else:
                    hit_purchase[i] += 1.0
                    ndcg_purchase[i] += 1.0 / np.log2(rank + 1)


def _concat(prefix, suffix, static=False):
  if isinstance(prefix, ops.Tensor):
    p = prefix
    p_static = tensor_util.constant_value(prefix)
    if p.shape.ndims == 0:
      p = array_ops.expand_dims(p, 0)
    elif p.shape.ndims != 1:
      raise ValueError("prefix tensor must be either a scalar or vector, "
                       "but saw tensor: %s" % p)
  else:
    p = tensor_shape.as_shape(prefix)
    p_static = p.as_list() if p.ndims is not None else None
    p = (constant_op.constant(p.as_list(), dtype=dtypes.int32)
         if p.is_fully_defined() else None)
  if isinstance(suffix, ops.Tensor):
    s = suffix
    s_static = tensor_util.constant_value(suffix)
    if s.shape.ndims == 0:
      s = array_ops.expand_dims(s, 0)
    elif s.shape.ndims != 1:
      raise ValueError("suffix tensor must be either a scalar or vector, "
                       "but saw tensor: %s" % s)
  else:
    s = tensor_shape.as_shape(suffix)
    s_static = s.as_list() if s.ndims is not None else None
    s = (constant_op.constant(s.as_list(), dtype=dtypes.int32)
         if s.is_fully_defined() else None)

  if static:
    shape = tensor_shape.as_shape(p_static).concatenate(s_static)
    shape = shape.as_list() if shape.ndims is not None else None
  else:
    if p is None or s is None:
      raise ValueError("Provided a prefix or suffix of None: %s and %s"
                       % (prefix, suffix))
    shape = array_ops.concat((p, s), 0)
  return shape

def _zero_state_tensors(state_size, batch_size, dtype):
  def get_state_shape(s):
    c = _concat(batch_size, s)
    size = array_ops.zeros(c, dtype=dtype)
    if not context.executing_eagerly():
      c_static = _concat(batch_size, s, static=True)
      size.set_shape(c_static)
    return size
  return nest.map_structure(get_state_shape, state_size)

class RNNCell(base_layer.Layer):

  def __init__(self, trainable=True, name=None, dtype=None, **kwargs):
    super(RNNCell, self).__init__(
        trainable=trainable, name=name, dtype=dtype, **kwargs)


    self._is_tf_rnn_cell = True

  def __call__(self, inputs, state, scope=None):
    if scope is not None:
      with vs.variable_scope(scope,
                             custom_getter=self._rnn_get_variable) as scope:
        return super(RNNCell, self).__call__(inputs, state, scope=scope)
    else:
      scope_attrname = "rnncell_scope"
      scope = getattr(self, scope_attrname, None)
      if scope is None:
        scope = vs.variable_scope(vs.get_variable_scope(),
                                  custom_getter=self._rnn_get_variable)
        setattr(self, scope_attrname, scope)
      with scope:
        return super(RNNCell, self).__call__(inputs, state)

  def _rnn_get_variable(self, getter, *args, **kwargs):
    variable = getter(*args, **kwargs)
    if context.executing_eagerly():
      trainable = variable._trainable
    else:
      trainable = (
          variable in tf_variables.trainable_variables() or
          (isinstance(variable, tf_variables.PartitionedVariable) and
           list(variable)[0] in tf_variables.trainable_variables()))
    if trainable and variable not in self._trainable_weights:
      self._trainable_weights.append(variable)
    elif not trainable and variable not in self._non_trainable_weights:
      self._non_trainable_weights.append(variable)
    return variable

  @property
  def state_size(self):
    raise NotImplementedError("Abstract method")

  @property
  def output_size(self):
    raise NotImplementedError("Abstract method")

  def build(self, _):


    pass

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    if inputs is not None:

      inputs = ops.convert_to_tensor(inputs, name="inputs")
      if batch_size is not None:
        if tensor_util.is_tensor(batch_size):
          static_batch_size = tensor_util.constant_value(
              batch_size, partial=True)
        else:
          static_batch_size = batch_size
        if inputs.shape.dims[0].value != static_batch_size:
          raise ValueError(
              "batch size from input tensor is different from the "
              "input param. Input tensor batch: {}, batch_size: {}".format(
                  inputs.shape.dims[0].value, batch_size))

      if dtype is not None and inputs.dtype != dtype:
        raise ValueError(
            "dtype from input tensor is different from the "
            "input param. Input tensor dtype: {}, dtype: {}".format(
                inputs.dtype, dtype))

      batch_size = inputs.shape.dims[0].value or array_ops.shape(inputs)[0]
      dtype = inputs.dtype
    if None in [batch_size, dtype]:
      raise ValueError(
          "batch_size and dtype cannot be None while constructing initial "
          "state: batch_size={}, dtype={}".format(batch_size, dtype))
    return self.zero_state(batch_size, dtype)

  def zero_state(self, batch_size, dtype):


    state_size = self.state_size
    is_eager = context.executing_eagerly()
    if is_eager and hasattr(self, "_last_zero_state"):
      (last_state_size, last_batch_size, last_dtype,
       last_output) = getattr(self, "_last_zero_state")
      if (last_batch_size == batch_size and
          last_dtype == dtype and
          last_state_size == state_size):
        return last_output
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      output = _zero_state_tensors(state_size, batch_size, dtype)
    if is_eager:
      self._last_zero_state = (state_size, batch_size, dtype, output)
    return output

class LayerRNNCell(RNNCell):

  def __call__(self, inputs, state, scope=None, *args, **kwargs):


    return base_layer.Layer.__call__(self, inputs, state, scope=scope,
                                     *args, **kwargs)

_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))

class LSTMStateTuple(_LSTMStateTuple):
  __slots__ = ()

  @property
  def dtype(self):
    (c, h) = self
    if c.dtype != h.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(c.dtype), str(h.dtype)))
    return c.dtype

class CLSTMCell(LayerRNNCell):

  @deprecated(None, "This class is equivalent as tf.keras.layers.LSTMCell,"
                    " and will be replaced by that in Tensorflow 2.0.")
  def __init__(self, num_units,
               use_peepholes=False, cell_clip=None,
               initializer=None, num_proj=None, proj_clip=None,
               num_unit_shards=None, num_proj_shards=None,
               forget_bias=1.0, state_is_tuple=True,
               activation=None, reuse=None, name=None, dtype=None, **kwargs):
    super(CLSTMCell, self).__init__(
        _reuse=reuse, name=name, dtype=dtype, **kwargs)
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    if num_unit_shards is not None or num_proj_shards is not None:
      logging.warn(
          "%s: The num_unit_shards and proj_unit_shards parameters are "
          "deprecated and will be removed in Jan 2017.  "
          "Use a variable scope with a partitioner instead.", self)
    if context.executing_eagerly() and context.num_gpus() > 0:
      logging.warn("%s: Note that this cell is not optimized for performance. "
                   "Please use tf.contrib.cudnn_rnn.CudnnLSTM for better "
                   "performance on GPU.", self)


    self.input_spec = input_spec.InputSpec(ndim=2)

    self._num_units = num_units
    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._initializer = initializers.get(initializer)
    self._num_proj = num_proj
    self._proj_clip = proj_clip
    self._num_unit_shards = num_unit_shards
    self._num_proj_shards = num_proj_shards
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    if activation:
      self._activation = activations.get(activation)
    else:
      self._activation = math_ops.tanh

    if num_proj:
      self._state_size = (
          LSTMStateTuple(num_units, num_proj)
          if state_is_tuple else num_units + num_proj)
      self._output_size = num_proj
    else:
      self._state_size = (
          LSTMStateTuple(num_units, num_units)
          if state_is_tuple else 2 * num_units)
      self._output_size = num_units

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  @tf_utils.shape_type_conversion
  def build(self, inputs_shape):
    if inputs_shape[-1] is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % str(inputs_shape))


    input_depth = inputs_shape[-1]
    h_depth = self._num_units if self._num_proj is None else self._num_proj
    maybe_partitioner = (
        partitioned_variables.fixed_size_partitioner(self._num_unit_shards)
        if self._num_unit_shards is not None
        else None)

    self._kernel = self.add_variable(
        _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + 2*h_depth, 4 * self._num_units],
        initializer=self._initializer,
        partitioner=maybe_partitioner)
    if self.dtype is None:
      initializer = init_ops.zeros_initializer
    else:
      initializer = init_ops.zeros_initializer(dtype=self.dtype)
    self._bias = self.add_variable(
        _BIAS_VARIABLE_NAME,
        shape=[4 * self._num_units],
        initializer=initializer)
    if self._use_peepholes:
      self._w_f_diag = self.add_variable("w_f_diag", shape=[self._num_units],
                                         initializer=self._initializer)
      self._w_i_diag = self.add_variable("w_i_diag", shape=[self._num_units],
                                         initializer=self._initializer)
      self._w_o_diag = self.add_variable("w_o_diag", shape=[self._num_units],
                                         initializer=self._initializer)

    if self._num_proj is not None:
      maybe_proj_partitioner = (
          partitioned_variables.fixed_size_partitioner(self._num_proj_shards)
          if self._num_proj_shards is not None
          else None)
      self._proj_kernel = self.add_variable(
          "projection/%s" % _WEIGHTS_VARIABLE_NAME,
          shape=[self._num_units, self._num_proj],
          initializer=self._initializer,
          partitioner=maybe_proj_partitioner)

    self.built = True

  def call(self, inputs, state):
    num_proj = self._num_units if self._num_proj is None else self._num_proj
    sigmoid = math_ops.sigmoid

    if self._state_is_tuple:
      (c_prev, m_prev) = state
    else:
      c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
      m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])
    print(1,inputs.shape)

    inputs,behaviors = array_ops.split(
        value=inputs, num_or_size_splits=2, axis=1)
    input_size = inputs.get_shape().with_rank(2).dims[1].value
    if input_size is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")


    lstm_matrix = math_ops.matmul(
        array_ops.concat([inputs, m_prev,behaviors,c_prev], 1), self._kernel)
    lstm_matrix = nn_ops.bias_add(lstm_matrix, self._bias)


    j_matrix = math_ops.matmul(
        array_ops.concat([inputs, m_prev,behaviors], 1), self._kernel[0:3*self._num_units])
    j_matrix = nn_ops.bias_add(j_matrix, self._bias)

    _, j, _, _ = array_ops.split(
        value = j_matrix, num_or_size_splits=4, axis=1)


    i, _, f, o = array_ops.split(
        value=lstm_matrix, num_or_size_splits=4, axis=1)

    if self._use_peepholes:
      c = (sigmoid(f + self._forget_bias + self._w_f_diag * c_prev) * c_prev +
           sigmoid(i + self._w_i_diag * c_prev) * self._activation(j))
    else:
      c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
           self._activation(j))

    if self._cell_clip is not None:

      c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)

    if self._use_peepholes:
      m = sigmoid(o + self._w_o_diag * c) * self._activation(c)
    else:
      m = sigmoid(o) * self._activation(c)

    if self._num_proj is not None:
      m = math_ops.matmul(m, self._proj_kernel)

      if self._proj_clip is not None:

        m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)


    new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else
                 array_ops.concat([c, m], 1))
    return m, new_state

  def get_config(self):
    config = {
        "num_units": self._num_units,
        "use_peepholes": self._use_peepholes,
        "cell_clip": self._cell_clip,
        "initializer": initializers.serialize(self._initializer),
        "num_proj": self._num_proj,
        "proj_clip": self._proj_clip,
        "num_unit_shards": self._num_unit_shards,
        "num_proj_shards": self._num_proj_shards,
        "forget_bias": self._forget_bias,
        "state_is_tuple": self._state_is_tuple,
        "activation": activations.serialize(self._activation),
        "reuse": self._reuse,
    }
    base_config = super(CLSTMCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
