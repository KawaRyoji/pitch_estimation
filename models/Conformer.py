from dataclasses import dataclass, field
from typing import Optional, Tuple

import tensorflow as tf

from deep_learning.dnn import DNN, ModelParams
from pitch_estimation.layers.common import GLU, RelativeMultiHeadAttention
from pitch_estimation.layers.embedding import PositionalEmbedding
from pitch_estimation.layers.util import shape_list


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
                dropout=0.1,
                output_dim=output_dim,
            )

    def get_model(self) -> Optional[tf.keras.Model]:
        return self.__model

    def definition(self, param: Params) -> tf.keras.Model:
        x = tf.keras.layers.Input(
            shape=param.input_size, name="input", dtype="float32"
        )  # (B, T, F)

        y = ConformerEncoder(
            dim=param.encoder_dim,
            num_layers=param.encoder_layers,
            num_heads=param.heads,
            kernel_size=param.kernel_size,
            dropout_rate=param.dropout,
        )(x)

        y = tf.keras.layers.Dense(param.output_dim, activation="sigmoid", name="output")(y)

        model = tf.keras.Model(inputs=x, outputs=y)

        return model


class ConformerEncoder(tf.keras.Model):
    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_heads: int,
        kernel_size: int,
        dropout_rate: float,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.fix_dim = tf.keras.layers.Dense(dim, name="fix-dim")
        self.pe = PositionalEmbedding(name="pos-emb")
        self.do = tf.keras.layers.Dropout(dropout_rate, name="pre-encoder-do")
        self.blocks = [
            ConformerBlock(
                dim,
                heads=num_heads,
                kernel_size=kernel_size,
                dropout=dropout_rate,
                name="conformer-block-%d" % i,
            )
            for i in range(num_layers)
        ]

    def call(self, inputs: tf.Tensor, training=None, mask=None) -> tf.Tensor:
        outputs = self.fix_dim(inputs)
        pe = self.pe(outputs)
        outputs = self.do(outputs)
        for block in self.blocks:
            outputs = block(outputs, pe=pe)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(self.fix_dim.get_config())
        config.update(self.pe.get_config())
        config.update(self.do.get_config())
        for block in self.blocks:
            config.update(block.get_config())

        return config


# 参考https://github.com/sooftware/conformer/blob/main/conformer
class ConformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        kernel_size: int = 32,
        dropout: float = 0.1,
        trainable=True,
        name=None,
        dtype=None,
        dynamic=False,
        **kwargs,
    ) -> None:
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.ff1 = FeedForwardModule(dim, dropout=dropout)
        self.mhsa = MultiHeadedSelfAttentionModule(dim, heads=heads)
        self.conv = ConvolutionModule(dim, kernel_size=kernel_size, dropout=dropout)
        self.ff2 = FeedForwardModule(dim, dropout=dropout)
        self.ln = tf.keras.layers.LayerNormalization()

    def call(
        self, inputs: tf.Tensor, pe: tf.Tensor, mask: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        # inputs (Batch, Length, Dim)
        outputs = self.ff1(inputs)
        outputs = self.mhsa(outputs, pe=pe, mask=mask)
        outputs = self.conv(outputs)
        outputs = self.ff2(outputs)
        return self.ln(outputs)

    def get_config(self):
        config = super().get_config()
        config.update(self.ff1.get_config())
        config.update(self.mhsa.get_config())
        config.update(self.conv.get_config())
        config.update(self.ff2.get_config())
        config.update(self.ln.get_config())
        return config


class FeedForwardModule(tf.keras.layers.Layer):
    def __init__(
        self,
        dim: int,
        rc_factor: float = 0.5,
        dropout: float = 0.1,
        trainable=True,
        name=None,
        dtype=None,
        dynamic=False,
        **kwargs,
    ) -> None:
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.rc_factor = rc_factor

        self.ln = tf.keras.layers.LayerNormalization()
        self.ffn1 = tf.keras.layers.Dense(4 * dim)
        self.swish = tf.keras.layers.Activation(tf.nn.swish)
        self.do1 = tf.keras.layers.Dropout(dropout)
        self.ffn2 = tf.keras.layers.Dense(dim)
        self.do2 = tf.keras.layers.Dropout(dropout)
        self.res_add = tf.keras.layers.Add()

    def call(self, inputs: tf.Tensor, training=False) -> tf.Tensor:
        outputs = self.ln(inputs, training=training)
        outputs = self.ffn1(outputs, training=training)
        outputs = self.swish(outputs)
        outputs = self.do1(outputs, training=training)
        outputs = self.ffn2(outputs, training=training)
        outputs = self.do2(outputs, training=training)
        outputs = self.res_add([inputs, self.rc_factor * outputs])
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({"rc_factor": self.rc_factor})
        config.update(self.ln.get_config())
        config.update(self.ffn1.get_config())
        config.update(self.swish.get_config())
        config.update(self.do1.get_config())
        config.update(self.ffn2.get_config())
        config.update(self.do2.get_config())
        config.update(self.res_add.get_config())
        return config


class MultiHeadedSelfAttentionModule(tf.keras.layers.Layer):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dropout: float = 0.1,
        trainable=True,
        name=None,
        dtype=None,
        dynamic=False,
        **kwargs,
    ) -> None:
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.ln = tf.keras.layers.LayerNormalization()
        self.attention = RelativeMultiHeadAttention(dim, heads=heads, dropout=dropout)
        self.do = tf.keras.layers.Dropout(dropout)
        self.res_add = tf.keras.layers.Add()

    def call(
        self, inputs: tf.Tensor, pe: tf.Tensor, mask: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        outputs = self.ln(inputs)
        outputs = self.attention(
            query=outputs, key=outputs, value=outputs, pos_emb=pe, mask=mask
        )
        outputs = self.do(outputs)
        return self.res_add([inputs, outputs])

    def get_config(self):
        config = super().get_config()
        config.update(self.ln.get_config())
        config.update(self.attention.get_config())
        config.update(self.do.get_config())
        config.update(self.res_add.get_config())
        return config


class ConvolutionModule(tf.keras.layers.Layer):
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
        self.ln = tf.keras.layers.LayerNormalization()  # (B, L, D)
        self.pw_conv1 = tf.keras.layers.Conv2D(
            filters=dim * 2,
            kernel_size=1,
            strides=1,
        )  # (B, L, D * 2)
        self.glu = GLU()  # (B, L, D) 以降同じ
        self.dw_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(kernel_size, 1),
            strides=1,
            padding="same",
        )
        self.bn = tf.keras.layers.BatchNormalization()
        self.swish = tf.keras.layers.Activation(tf.nn.swish)
        self.pw_conv2 = tf.keras.layers.Conv2D(
            filters=dim,
            kernel_size=1,
            strides=1,
        )
        self.do = tf.keras.layers.Dropout(dropout)
        self.res_add = tf.keras.layers.Add()

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        outputs = self.ln(inputs)
        B, L, D = shape_list(outputs)
        outputs = tf.reshape(outputs, [B, L, 1, D])
        outputs = self.pw_conv1(outputs)
        outputs = self.glu(outputs)
        outputs = self.dw_conv(outputs)
        outputs = self.bn(outputs)
        outputs = self.swish(outputs)
        outputs = self.pw_conv2(outputs)
        outputs = tf.reshape(outputs, [B, L, D])
        outputs = self.do(outputs)
        return self.res_add([inputs, outputs])

    def get_config(self):
        config = super().get_config()
        config.update(self.ln.get_config())
        config.update(self.pw_conv1.get_config())
        config.update(self.glu.get_config())
        config.update(self.dw_conv.get_config())
        config.update(self.bn.get_config())
        config.update(self.swish.get_config())
        config.update(self.pw_conv2.get_config())
        config.update(self.do.get_config())
        config.update(self.res_add.get_config())
        return config
