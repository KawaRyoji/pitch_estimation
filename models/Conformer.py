from dataclasses import dataclass, field
from typing import Optional, Tuple

import tensorflow as tf
from deep_learning.dnn import DNN, ModelParams
from pitch_estimation.layers.common import (
    GLU,
    DepthwiseConv1d,
    RelativeMultiHeadAttention,
)
from pitch_estimation.layers.embedding import PositionalEmbedding
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Input,
    Layer,
    LayerNormalization,
)


class Conformer(DNN):
    @dataclass
    class Params(ModelParams):
        input_size: Tuple[int, int] = field(default=(128, 1024))
        encoder_layers: int = field(default=16)
        encoder_dim: int = field(default=256)
        heads: int = field(default=4)
        ffn_expansion_factor: int = field(default=4)
        kernel_size: int = field(default=32)
        dropout: float = field(default=0.1)
        output_dim: int = field(default=128)

        @classmethod
        def small(cls, input_size=(128, 1024), output_dim=128) -> "Conformer.Params":
            return cls(
                input_size=input_size,
                encoder_layers=16,
                encoder_dim=144,
                heads=4,
                kernel_size=31,
                ffn_expansion_factor=4,
                dropout=0.1,
                output_dim=output_dim,
            )

        @classmethod
        def medium(cls, input_size=(128, 1024), output_dim=128) -> "Conformer.Params":
            return cls(
                input_size=input_size,
                encoder_layers=16,
                encoder_dim=256,
                heads=4,
                kernel_size=31,
                ffn_expansion_factor=4,
                dropout=0.1,
                output_dim=output_dim,
            )

        @classmethod
        def large(cls, input_size=(128, 1024), output_dim=128) -> "Conformer.Params":
            return cls(
                input_size=input_size,
                encoder_layers=17,
                encoder_dim=512,
                heads=8,
                kernel_size=31,
                ffn_expansion_factor=4,
                dropout=0.1,
                output_dim=output_dim,
            )

    def definition(self, param: Params) -> Model:
        x = Input(shape=param.input_size, name="input", dtype="float32")  # (B, T, F)

        y = Dense(param.encoder_dim, name="fix_dim")(x)  # 入力の次元をEncoder Dim に調整
        y = Dropout(param.dropout, name="input-do")(y)

        for i in range(param.encoder_layers):
            y = ConformerBlock(
                dim=param.encoder_dim,
                heads=param.heads,
                ffn_expansion_factor=param.ffn_expansion_factor,
                kernel_size=param.kernel_size,
                dropout=param.dropout,
                name="conformer%d" % i,
            )(y)

        y = Dense(param.output_dim, activation="sigmoid", name="output")(y)

        model = Model(inputs=x, outputs=y)

        return model


# 参考https://github.com/sooftware/conformer/blob/main/conformer
class ConformerBlock(Layer):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        ffn_expansion_factor: int = 4,
        kernel_size: int = 31,
        dropout: float = 0.1,
        max_length: int = 10000,
        trainable=True,
        name=None,
        dtype=None,
        dynamic=False,
        **kwargs,
    ) -> None:
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.ff1 = FeedForwardModule(
            dim, expansion_factor=ffn_expansion_factor, dropout=dropout
        )
        self.mhsa = MultiHeadedSelfAttentionModule(
            dim, heads=heads, dropout=dropout, max_length=max_length
        )
        self.conv = ConvolutionModule(dim, kernel_size=kernel_size, dropout=dropout)
        self.ff2 = FeedForwardModule(
            dim, expansion_factor=ffn_expansion_factor, dropout=dropout
        )
        self.ln = LayerNormalization()

    def call(self, inputs: tf.Tensor, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        # inputs (Batch, Length, Dim)
        inputs = self.ff1(inputs) * 0.5 + inputs
        inputs = self.mhsa(inputs, mask=mask) + inputs
        inputs = self.conv(inputs) + inputs
        inputs = self.ff2(inputs) * 0.5 + inputs
        return self.ln(inputs)


class FeedForwardModule(Layer):
    def __init__(
        self,
        dim: int,
        expansion_factor: int = 4,
        dropout: float = 0.1,
        trainable=True,
        name=None,
        dtype=None,
        dynamic=False,
        **kwargs,
    ) -> None:
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.net = Sequential(
            [
                LayerNormalization(),
                Dense(dim * expansion_factor, activation="swish"),
                Dropout(dropout),
                Dense(dim),
                Dropout(dropout),
            ]
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.net(inputs)


class MultiHeadedSelfAttentionModule(Layer):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dropout: float = 0.1,
        max_length: int = 10000,
        trainable=True,
        name=None,
        dtype=None,
        dynamic=False,
        **kwargs,
    ) -> None:
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.positional_emb = PositionalEmbedding(dim, max_len=max_length)
        self.ln = LayerNormalization()
        self.attention = RelativeMultiHeadAttention(dim, heads=heads, dropout=dropout)
        self.do = Dropout(dropout)

    def call(self, inputs: tf.Tensor, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        batch_size, length, _ = tf.unstack(tf.shape(inputs))
        pe = tf.tile(
            tf.expand_dims(self.positional_emb(length), axis=0), [batch_size, 1, 1]
        )

        inputs = self.ln(inputs)
        inputs = self.attention(
            query=inputs, key=inputs, value=inputs, pos_emb=pe, mask=mask
        )

        return self.do(inputs)


class ConvolutionModule(Layer):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 31,  # 論文では32だが、カーネルサイズは奇数が基本のため31
        dropout: float = 0.1,
        trainable=True,
        name=None,
        dtype=None,
        dynamic=False,
        **kwargs,
    ) -> None:
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.net = Sequential(
            [
                LayerNormalization(),  # (B, L, D)
                Conv1D(
                    filters=dim * 2,
                    kernel_size=1,
                    strides=1,
                ),  # (B, L, D * 2)
                GLU(axis=-1),  # (B, L, D) 以降同じ
                DepthwiseConv1d(
                    dim=dim,
                    kernel_size=kernel_size,
                    padding="same",
                ),
                BatchNormalization(),
                Activation("swish"),
                Conv1D(filters=dim, kernel_size=1),
                Dropout(dropout),
            ]
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.net(inputs)
