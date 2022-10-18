from dataclasses import dataclass, field

from deep_learning.dnn import DNN, ModelParams
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    LayerNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPool2D,
    Permute,
    Reshape,
    SeparableConv2D
)
import tensorflow as tf


class Conformer(DNN):
    @dataclass
    class Params(ModelParams):
        pass

    def definition(self, param: ModelParams) -> Model:
        return super().definition(param)

    def conformer_block(self, x: tf.Tensor, layer: int) -> tf.Tensor:
        x = LayerNormalization(name="conv-LN%d" % layer)(x)
        Conv2D()

    def feed_formward_module(self, x: tf.Tensor, layer: int) -> tf.Tensor:
        pass

    def multi_head_self_attention_module(self, x: tf.Tensor, layer: int) -> tf.Tensor:
        pass

    def convolution_module(self, x: tf.Tensor, layer: int) -> tf.Tensor:
        pass
