from ast import Lambda
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer, Lambda


class PositionalEmbedding(Layer):
    def __init__(
        self,
        dim: int,
        max_len=10000,
        trainable=True,
        name=None,
        dtype=tf.float32,
        dynamic=False,
        **kwargs
    ) -> None:
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.dim = dim
        self.max_len = max_len
        self.pe = tf.Variable(self.__create(), name="positional_embedding", dtype=dtype)

    def call(self, length: int) -> tf.Tensor:
        return self.pe[:length, :]

    def __create(self) -> tf.Tensor:
        # NOTE: expend_dims(x, 0)とすると[x]から[1, shape(x)]となる
        #       これに加えてtile(expand_dims(x, 0), [nums, 1])とすると[nums, shape(x)]となり
        #       テンソルxを縦にnums回並べたテンソルとなる

        dim_counter = tf.range(self.dim) // 2 * 2  # 0, 0, 2, 2, 4, ...
        dim_matrix = tf.tile(tf.expand_dims(dim_counter, 0), [self.max_len, 1])
        dim_matrix = tf.pow(10000.0, tf.cast(dim_matrix / self.dim, self.dtype))

        phase = (
            tf.cast(tf.range(self.dim) % 2, self.dtype) % np.pi / 2
        )  # 0, pi/2, 0, pi/2, ...
        phase_matrix = tf.tile(tf.expand_dims(phase, 0), [self.max_len, 1])

        pos_counter = tf.range(self.max_len)
        pos_matrix = tf.cast(
            tf.tile(tf.expand_dims(pos_counter, 1), [1, self.dim]), self.dtype
        )

        pe = tf.sin(pos_matrix / dim_matrix + phase_matrix)
        return pe
