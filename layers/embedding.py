import tensorflow as tf
from tensorflow.keras.layers import Layer

from pitch_estimation.layers.util import shape_list


class PositionalEmbedding(Layer):
    def __init__(
        self, trainable=True, name=None, dtype=tf.float32, dynamic=False, **kwargs
    ) -> None:
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        batch_size, max_len, dim = shape_list(inputs)
        pe = self.encode(max_len, dim)
        pe = tf.tile(pe, [batch_size, 1, 1])
        return tf.cast(pe, dtype=inputs.dtype)

    @staticmethod
    def encode(max_len: int, dim: int) -> tf.Tensor:
        dim_f = tf.cast(dim, dtype=tf.float32)
        pos = tf.expand_dims(tf.range(max_len - 1, -1, -1.0, dtype=tf.float32), axis=1)
        index = tf.expand_dims(tf.range(0, dim, dtype=tf.float32), axis=0)

        pe = pos * (1 / tf.pow(10000.0, (2 * (index // 2)) / dim_f))

        # Sin cos will be [max_len, size // 2]
        # we add 0 between numbers by using padding and reshape
        sin = tf.pad(
            tf.expand_dims(tf.sin(pe[:, 0::2]), -1),
            [[0, 0], [0, 0], [0, 1]],
            mode="CONSTANT",
            constant_values=0,
        )
        sin = tf.reshape(sin, [max_len, dim])
        cos = tf.pad(
            tf.expand_dims(tf.cos(pe[:, 1::2]), -1),
            [[0, 0], [0, 0], [1, 0]],
            mode="CONSTANT",
            constant_values=0,
        )
        cos = tf.reshape(cos, [max_len, dim])
        # Then add sin and cos, which results in [time, size]
        pe = tf.add(sin, cos)
        return tf.expand_dims(pe, axis=0)  # [1, time, size]

    def get_config(self):
        return super().get_config()
