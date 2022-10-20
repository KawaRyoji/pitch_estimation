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
    Flatten,
    Input,
    MaxPool2D,
    Permute,
    Reshape,
    SeparableConv2D,
)
import tensorflow as tf


class Conformer(DNN):
    @dataclass
    class Params(ModelParams):
        encoder_layers: int = field(default=16)
        encoder_dim: int = field(default=256)
        attention_heads: int = field(default=4)
        kernel_size: int = field(default=32)
        dropout_rate: float = field(default=0.1)

    def definition(self, param: ModelParams) -> Model:
        return super().definition(param)

    def conformer_block(self, x: tf.Tensor, layer: int) -> tf.Tensor:
        y1 = self.feed_forward_module(x, 1, layer)(x)
        y1 = tf.math.scalar_mul(0.5, y1, name="ffn1-half_%d" % layer)
        y1 = Add(name="ffn1-Add_%d" % layer)((x, y1))
        y2 = self.multi_head_self_attention_module(x, layer)(y1)
        y2 = Add(name="mha-Add_%d" % layer)((y1, y2))
        y3 = self.convolution_module(x, layer)(y2)
        y3 = Add(name="conv-Add_%d" % layer)((y2, y3))
        y4 = self.feed_forward_module(x, 1, layer)(y3)
        y4 = tf.math.scalar_mul(0.5, y4, name="ffn1-half_%d" % layer)
        y4 = Add(name="ffn2-Add_%d" % layer)((y3, y4))
        y4 = LayerNormalization(name="conf-LN_%d" % layer)(y4)

        return y4

    def feed_forward_module(self, x: tf.Tensor, num: int, layer: int) -> tf.Tensor:
        y = LayerNormalization(name="ffn-LN_%d" % layer)(x)
        y = Dense(name="ffn%d-D1_%d" % (num, layer), activation="swish")(y)
        y = Dropout(name="ffn%d-DO1_%d" % (num, layer))(y)
        y = Dense(name="ffn%d-D2_%d" % (num, layer))(y)
        y = Dropout(name="ffn%d-DO2_%d" % (num, layer))(y)

        return y

    def multi_head_self_attention_module(self, x: tf.Tensor, layer: int) -> tf.Tensor:
        y = LayerNormalization(name="mha-LN_%d" % layer)(x)
        y = MultiHeadAttention(name="mha-MHA_%d" % layer)(y)
        y = Dropout(name="mha-DO_%d" % layer)(y)

        return y

    def convolution_module(self, x: tf.Tensor, layer: int) -> tf.Tensor:
        y = LayerNormalization(name="conv-LN_%d" % layer)(x)
        y = Conv2D(name="conv-DwC_%d" % layer)(y)
        y = GLU(name="conv-GLU")(y)
        y = BatchNormalization(name="conv-BN_%d" % layer)(y)
        y = Conv2D(name="conv-PwC_%d" % layer)(y)
        y = Dropout(name="conv-DO_%d" % layer)(y)

        return y
