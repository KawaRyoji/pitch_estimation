from typing import Optional
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Layer, Conv1D


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


class RelativeMultiHeadAttention(Layer):
    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float,
        trainable=True,
        name=None,
        dtype=None,
        dynamic=False,
        **kwargs
    ) -> None:
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.dim = dim
        self.dim_head = dim // heads
        self.heads = heads

        self.q_linear = Dense(dim)
        self.k_linear = Dense(dim)
        self.v_linear = Dense(dim)
        self.pos_linear = Dense(dim, use_bias=False)
        self.o_linear = Dense(dim)
        self.do = Dropout(dropout)

    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        pos_emb: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        batch_size, _, _ = tf.unstack(tf.shape(query))

        query = self._split_head(self.q_linear(query), batch_size)  # (B, H, QL, HD)
        key = self._split_head(self.k_linear(key), batch_size)  # (B, H, KL, HD)
        value = self._split_head(self.v_linear(value), batch_size)  # (B, H, VL, HD)
        pos_emb = self._split_head(
            self.pos_linear(pos_emb), batch_size
        )  # (B, H, PL, HD)

        content_logit = tf.matmul(query, key, transpose_b=True)  # (B, H, QL, KL)
        pos_logit = tf.matmul(query, pos_emb, transpose_b=True)  # (B, H, QL, PL)
        pos_logit = self._relative_shift(pos_logit)

        logit: tf.Tensor = (content_logit + pos_logit) * (
            self.dim**-0.5
        )  # (B, H, QL, KL)

        if mask is not None:
            logit = tf.where(mask, logit.dtype.min, logit)

        attention_score = tf.nn.softmax(logit)
        attention_score = self.do(attention_score)  # (B, H, QL, KL)

        context = tf.matmul(attention_score, value)  # (B, H, QL, HD)
        context = tf.reshape(context, [batch_size, -1, self.dim])  # (B, QL, D)

        return self.o_linear(context)

    def _split_head(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
        x = tf.reshape(x, (batch_size, -1, self.heads, self.dim_head))
        return tf.transpose(x, [0, 2, 1, 3])

    # Skew アルゴリズム
    # https://jaketae.github.io/study/relative-positional-encoding/
    def _relative_shift(self, pos_score: tf.Tensor) -> tf.Tensor:
        # (B, H, L, L)
        pos_score = tf.pad(
            pos_score, [[0, 0], [0, 0], [0, 0], [1, 0]]
        )  # (B, H, L, 1 + L)
        pos_score = tf.transpose(pos_score, [0, 1, 3, 2])  # (B, H, 1 + L, L)
        return pos_score[:, :, 1:]  # (B, H, L, L)


class GLU(Layer):
    def __init__(
        self, axis: int, trainable=True, name=None, dtype=None, dynamic=False, **kwargs
    ) -> None:
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.axis = axis

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        out, gate = tf.split(inputs, 2, axis=self.axis)
        return out * tf.sigmoid(gate)


class DepthwiseConv1d(Layer):
    def __init__(
        self,
        dim: int,
        kernel_size: int,
        stride: int = 1,
        padding="valid",
        trainable=True,
        name=None,
        dtype=None,
        dynamic=False,
        **kwargs
    ) -> None:
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.conv = Conv1D(
            filters=dim,
            kernel_size=kernel_size,
            groups=dim,
            strides=stride,
            padding=padding,
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.conv(inputs)
