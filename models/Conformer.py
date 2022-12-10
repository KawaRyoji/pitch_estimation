from dataclasses import dataclass, field
from typing import Tuple

from deep_learning.dnn import DNN, ModelParams
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from conformer_tf import ConformerBlock


class Conformer(DNN):
    @dataclass
    class Params(ModelParams):
        input_size: Tuple[int, int] = field(default=(1024, 128))
        encoder_layers: int = field(default=16)
        encoder_dim: int = field(default=256)
        attention_heads: int = field(default=4)
        ffn_inner_dim_factor: int = field(default=4)
        conv_inner_dim_factor: int = field(default=2)
        kernel_size: int = field(default=32)
        dropout_rate: float = field(default=0.1)
        output_dim: int = field(default=128)

        @classmethod
        def small(cls, input_size=(1024, 128), output_dim=128) -> "Conformer.Params":
            return cls(
                input_size=input_size,
                encoder_layers=16,
                encoder_dim=144,
                attention_heads=4,
                kernel_size=32,
                ffn_inner_dim_factor=4,
                conv_inner_dim_factor=2,
                dropout_rate=0.1,
                output_dim=output_dim,
            )

        @classmethod
        def medium(cls, input_size=(1024, 128), output_dim=128) -> "Conformer.Params":
            return cls(
                input_size=input_size,
                encoder_layers=16,
                encoder_dim=256,
                attention_heads=4,
                kernel_size=32,
                ffn_inner_dim_factor=4,
                conv_inner_dim_factor=2,
                dropout_rate=0.1,
                output_dim=output_dim,
            )

        @classmethod
        def large(cls, input_size=(1024, 128), output_dim=128) -> "Conformer.Params":
            return cls(
                input_size=input_size,
                encoder_layers=17,
                encoder_dim=512,
                attention_heads=8,
                kernel_size=32,
                ffn_inner_dim_factor=4,
                conv_inner_dim_factor=2,
                dropout_rate=0.1,
                output_dim=output_dim,
            )

    def definition(self, param: Params) -> Model:
        x = Input(shape=param.input_size, name="input", dtype="float32")

        y = Dense(param.encoder_dim, name="fix_dim")(x)  # 入力の次元をEncoder Dim に調整
        y = Dropout(param.dropout_rate, name="input-do")(y)

        for i in range(param.encoder_layers):
            y = ConformerBlock(
                dim=param.encoder_dim,
                dim_head=param.encoder_dim // param.attention_heads,
                heads=param.attention_heads,
                ff_mult=param.ffn_inner_dim_factor,
                conv_expansion_factor=2,
                conv_kernel_size=param.kernel_size,
                attn_dropout=param.dropout_rate,
                ff_dropout=param.dropout_rate,
                conv_dropout=param.dropout_rate,
                name="conformer%d" % i,
            )(y)

        y = Dense(param.output_dim, activation="sigmoid", name="output")(y)

        model = Model(inputs=x, outputs=y)

        return model
