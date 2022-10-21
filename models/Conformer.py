from dataclasses import dataclass, field

from deep_learning.dnn import DNN, ModelParams
from pitch_estimation.layers.common import GLU
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    LayerNormalization,
    Conv2D,
    BatchNormalization,
    Add,
    MultiHeadAttention,
    Dense,
    Dropout,
    Layer,
    Input,
    MaxPool2D,
    Permute,
    Reshape,
)
import tensorflow as tf


class Conformer(DNN):
    @dataclass
    class Params(ModelParams):
        encoder_layers: int = field(default=16)
        encoder_dim: int = field(default=256)
        ffn_inner_dim_factor: int = field(default=4)
        attention_heads: int = field(default=4)
        kernel_size: int = field(default=32)
        dropout_rate: float = field(default=0.1)

    def definition(self, param: Params) -> Model:
        return super().definition(param)


class FeedForwardModule(Layer):
    def __init__(
        self,
        layer_num: int,
        input_dim: int,
        inner_dim_factor: int = 4,
        dropout_rate: float = 0.1,
        scale_factor: float = 0.5,
        trainable=True,
        name="ffn",
        dtype=None,
        dynamic=False,
        **kwargs
    ) -> None:
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

        self.inner_dim_factor = inner_dim_factor
        self.scale_factor = scale_factor

        self.ln = LayerNormalization(name="%s-LN-%d" % (name, layer_num))
        self.ffn1 = Dense(
            input_dim * inner_dim_factor,
            name="%s-in-D1-%d" % (name, layer_num),
            activation="swish",
        )
        self.do1 = Dropout(dropout_rate, name="%s-DO1-%d" % (name, layer_num))
        self.ffn2 = Dense(name="%s-out-D2-%d" % (name, layer_num))
        self.do2 = Dropout(dropout_rate, name="%s-DO2-%d" % (name, layer_num))
        self.res_add = Add(name="%s-Add-%d" % (name, layer_num))

    def call(self, inputs, training=False) -> tf.Tensor:
        outputs = self.ln(inputs, training=training)
        outputs = self.ffn1(outputs, training=training)
        outputs = self.do1(outputs, training=training)
        outputs = self.ffn2(outputs, training=training)
        outputs = self.do2(outputs, training=training)
        outputs = self.res_add([inputs, self.scale_factor * outputs])

        return outputs


class ConvolutionModule(Layer):
    def __init__(
        self,
        layer_num: int,
        trainable=True,
        name="conv",
        dtype=None,
        dynamic=False,
        **kwargs
    ) -> None:
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

        self.ln = LayerNormalization(name="%s-LN-%d" % (name, layer_num))
        self.pw_conv = Conv2D(name="%s-DwC-%d" % (name, layer_num))
        self.glu = GLU(name="%s-GLU-%d" % (name, layer_num))
        self.bn = BatchNormalization(name="%s-BN-%d" % (name, layer_num))
        self.dw_conv = Conv2D(name="%s-PwC-%d" % (name, layer_num))
        self.do = Dropout(name="%s-DO-%d" % (name, layer_num))
        self.res_add = Add(name="%s-Add-%d" % (name, layer_num))

    def call(self, inputs, *args, **kwargs) -> tf.Tensor:
        return super().call(inputs, *args, **kwargs)


class MultiHeadAttentionModule(Layer):
    def __init__(
        self,
        layer_num: int,
        trainable=True,
        name="mha",
        dtype=None,
        dynamic=False,
        **kwargs
    ) -> None:
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

        self.ln = LayerNormalization(name="%s-LN-%d" % (name, layer_num))
        self.rel_mha = MultiHeadAttention(name="%s-MHA-%d" % (name, layer_num))
        self.do = Dropout(name="%s-DO-%d" % (name, layer_num))
        self.res_add = Add(name="%s-Add-%d" % (name, layer_num))

    def call(self, inputs, *args, **kwargs) -> tf.Tensor:
        return super().call(inputs, *args, **kwargs)


class ConformerModule(Layer):
    def __init__(
        self,
        layer_num: int,
        trainable=True,
        name="conf",
        dtype=None,
        dynamic=False,
        **kwargs
    ):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

        self.ffn1 = FeedForwardModule()
        self.mha = MultiHeadAttentionModule()
        self.conv = ConvolutionModule()
        self.ffn2 = FeedForwardModule()
        self.ln = LayerNormalization(name="%s-LN-%d" % (name, layer_num))

    def call(self, inputs, *args, **kwargs) -> tf.Tensor:
        return super().call(inputs, *args, **kwargs)
