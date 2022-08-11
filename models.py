from dataclasses import dataclass, field
import tensorflow as tf
from deep_learning.dnn import DNN, ModelParams
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Conv1D,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    Input,
    Lambda,
    Layer,
    MaxPool2D,
    MaxPooling1D,
    Multiply,
    Permute,
    Reshape,
)
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

from pitch_estimation.layers.transformer import Encoder


class CREPE(DNN):
    @dataclass
    class Params(ModelParams):
        input_size: int = field(default=1024)
        first_stride: int = field(default=4)

    def definition(self, params: Params) -> Model:
        input_size = params.input_size
        first_stride = params.first_stride

        capacity_multiplier = 32
        layers = [1, 2, 3, 4, 5, 6]
        filters = [n * capacity_multiplier for n in [32, 4, 4, 4, 8, 16]]
        widths = [
            input_size // 2,
            input_size // 16,
            input_size // 16,
            input_size // 16,
            input_size // 16,
            input_size // 16,
        ]
        strides = [(first_stride, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

        x = Input(shape=(input_size,), name="input", dtype="float32")
        y = Reshape(target_shape=(input_size, 1, 1), name="input-reshape")(x)

        for l, f, w, s in zip(layers, filters, widths, strides):
            y = Conv2D(
                f,
                (w, 1),
                strides=s,
                padding="same",
                activation="relu",
                name="conv%d" % l,
            )(y)
            y = BatchNormalization(name="conv%d-BN" % l)(y)
            y = MaxPool2D(
                pool_size=(2, 1),
                strides=None,
                padding="valid",
                name="conv%d-maxpool" % l,
            )(y)
            y = Dropout(0.25, name="conv%d-dropout" % l)(y)

        y = Permute((2, 1, 3), name="transpose")(y)
        y = Flatten(name="flatten")(y)
        y = Dense(128, activation="sigmoid", name="classifier")(y)

        model = Model(inputs=x, outputs=y)

        return model


class DA_Net(DNN):
    @dataclass
    class Params(ModelParams):
        input_size: int = field(default=1024)

    def definition(self, params: Params) -> Model:
        input_size = params.input_size

        layers = [1, 2, 3, 4, 5, 6]
        kernel_size = [512, 64, 64, 64, 64, 64]
        filters = [1024, 128, 128, 128, 256, 512]
        strides = [4, 1, 1, 1, 1, 1]
        pooling_size = 2

        x = Input(shape=(input_size,), name="input", dtype="float32")
        y = Reshape(target_shape=(input_size, 1), name="input-reshape")(x)

        for l, f, k, s in zip(layers, filters, kernel_size, strides):
            y = self.__da_module(y, l, f, k, s, pooling_size)

        y = Flatten(name="flatten")(y)
        y = Dense(128, activation="sigmoid", name="classifier")(y)

        model = Model(inputs=x, outputs=y)

        return model

    def __da_module(
        self,
        x: tf.Tensor,
        layer: int,
        filter: int,
        kernel_size: int,
        stride: int,
        pooling_size: int,
    ) -> Model:
        r = 16

        # Element-wise Attention
        y_ew_u = Conv1D(
            filter,
            kernel_size,
            strides=stride,
            padding="same",
            name="da-u-conv%d" % layer,
        )(x)
        y_ew_b = Conv1D(
            filter,
            kernel_size,
            strides=stride,
            padding="same",
            activation="sigmoid",
            name="da-b-conv%d" % layer,
        )(x)
        y_ew = Multiply(name="da-ew-mul%d" % layer)([y_ew_u, y_ew_b])

        # Channel-wise Attention
        y_cw = GlobalAveragePooling1D(name="da-cw-gap%d" % layer)(y_ew_u)
        y_cw = Dense(filter // r, activation="relu", name="da-cw-dr%d" % layer)(y_cw)
        y_cw = Dense(filter, activation="sigmoid", name="da-cw-ds%d" % layer)(y_cw)
        y_cw = Multiply(name="da-cw-mul%d" % layer)([y_ew_u, y_cw])

        y = Add(name="da-add%d" % layer)([y_ew, y_cw])
        y = BatchNormalization(name="da-BN%d" % layer)(y)
        y = MaxPooling1D(pool_size=pooling_size, name="da-mp%d" % layer)(y)
        y = Dropout(0.25, name="da-do%d" % layer)(y)

        return y


class Transformer(DNN):
    @dataclass
    class Params(ModelParams):
        encoder_input_dim: int = field(default=1024)
        output_dim: int = field(default=128)
        data_length: int = field(default=16)
        decoder: Layer = field(default=Dense(128))
        hidden_dim: int = field(default=512)
        ffn_dim: int = field(default=2048)
        num_layer: int = field(default=6)
        num_head: int = field(default=8)
        dropout_rate: float = field(default=0.1)

    def definition(self, params: Params) -> Model:
        input = Input(
            shape=(
                params.data_length,
                params.encoder_input_dim,
            )
        )

        mask = Lambda(self._create_enc_self_attention_mask)(input)

        y = Encoder(
            params.hidden_dim,
            params.num_layer,
            params.num_head,
            params.ffn_dim,
            params.dropout_rate,
        )(input, mask=mask)

        y = params.decoder(y)

        output = Dense(params.output_dim, activation="sigmoid")(y)

        model = Model(inputs=input, outputs=output)

        return model

    # NOTE: <PAD>を用いない実験のためマスクを作成していない
    def _create_enc_self_attention_mask(self, input: tf.Tensor) -> tf.Tensor:
        batch_size, length, _ = tf.unstack(tf.shape(input))

        array = tf.zeros([batch_size, length], dtype=tf.bool)
        return tf.reshape(array, [batch_size, 1, 1, length])


class TransformerLearningRateScheduler(LearningRateSchedule):
    def __init__(self, max_learning_rate=0.0001, warmup_step=4000) -> None:
        self.max_learning_rate = max_learning_rate
        self.warmup_step = warmup_step

    def __call__(self, step) -> float:
        rate = (
            tf.minimum(step**-0.5, step * self.warmup_step**-1.5)
            / self.warmup_step**-0.5
        )
        return self.max_learning_rate * rate
