import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Layer


class MultiHeadAttention(Layer):
    def __init__(
        self, dim: int, head_num: int, dropout_rate: float, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.head_num = head_num
        self.dropout_rate = dropout_rate

        self.q_dense = Dense(dim, use_bias=False, name="q_dense_layer")
        self.k_dense = Dense(dim, use_bias=False, name="k_dense_layer")
        self.v_dense = Dense(dim, use_bias=False, name="v_dense_layer")
        self.o_dense = Dense(dim, use_bias=False, name="output_dense_layer")
        self.attention_dropout = Dropout(dropout_rate)

    def call(self, input: tf.Tensor, memory: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        q = self.q_dense(input)
        k = self.k_dense(memory)
        v = self.v_dense(memory)

        q = self._split_head(q)
        k = self._split_head(k)
        v = self._split_head(v)

        depth = self.dim // self.head_num
        q: tf.Tensor = q * depth**-0.5

        logit = tf.matmul(q, k, transpose_b=True)
        logit += tf.cast(mask, q.dtype) * q.dtype.min

        attention_score = tf.nn.softmax(logit, name="attention_score")
        attention_score = self.attention_dropout(attention_score)

        attention_output = tf.matmul(attention_score, v)
        attention_output = self._concat_head(attention_output)

        return self.o_dense(attention_output)

    def _split_head(self, x: tf.Tensor) -> tf.Tensor:
        with tf.name_scope("split_head"):
            batch_size, length, _ = tf.unstack(tf.shape(x))
            x = tf.reshape(
                x, [batch_size, length, self.head_num, self.dim // self.head_num]
            )
            return tf.transpose(x, [0, 2, 1, 3])

    def _concat_head(self, x: tf.Tensor) -> tf.Tensor:
        with tf.name_scope("concat_head"):
            batch_size, _, length, _ = tf.unstack(tf.shape(x))
            x = tf.transpose(x, [0, 2, 1, 3])
            return tf.reshape(x, [batch_size, length, self.dim])


class SelfAttention(MultiHeadAttention):
    def call(self, input: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        return super().call(input, memory=input, mask=mask)
