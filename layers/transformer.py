import math

import tensorflow as tf
from pitch_estimation.layers.common import MultiHeadAttention, SelfAttention
from tensorflow.keras.layers import (
    Add,
    Dense,
    Dropout,
    Lambda,
    Layer,
    LayerNormalization,
)


class Decoder(Layer):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        num_layer: int,
        num_head: int,
        ffn_dim: int,
        dropout_rate: float,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layer = num_layer
        self.num_head = num_head
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout_rate

        self.token_embedding = TokenEmbedding(hidden_dim)
        self.add_position_embedding = AddPositionalEncoding()
        self.dropout = Dropout(dropout_rate)

        self.dec_blocks = []
        for _ in range(num_layer):
            self_attention_layer = SelfAttention(hidden_dim, num_head, dropout_rate)
            enc_dec_attention_layer = MultiHeadAttention(
                hidden_dim, num_head, dropout_rate
            )
            ffn_layer = FeedForwardNetwork(hidden_dim, ffn_dim, dropout_rate)

            self.dec_blocks.append(
                [
                    ResidualNormalizationWrapper(self_attention_layer, dropout_rate),
                    ResidualNormalizationWrapper(enc_dec_attention_layer, dropout_rate),
                    ResidualNormalizationWrapper(ffn_layer, dropout_rate),
                ]
            )

        self.layer_normalization = LayerNormalization()
        self.dense = Dense(output_dim)

    def call(
        self,
        input: tf.Tensor,
        encoder_output: tf.Tensor,
        self_attention_mask: tf.Tensor,
        enc_dec_attention_mask: tf.Tensor,
    ) -> tf.Tensor:
        embedded_input = self.token_embedding(input)
        embedded_input = self.add_position_embedding(embedded_input)
        query = self.dropout(embedded_input)

        for i, layers in enumerate(self.dec_blocks):
            self_attention_layer, enc_dec_attention_layer, ffn_layer = tuple(layers)

            query = self_attention_layer(query, mask=self_attention_mask)
            query = enc_dec_attention_layer(
                query, memory=encoder_output, mask=enc_dec_attention_mask
            )
            query = ffn_layer(query)

        query = self.layer_normalization(query)
        return self.dense(query)


class TokenEmbedding(Layer):
    def __init__(self, hidden_dim: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.dense = Dense(hidden_dim, use_bias=False)

    def call(self, input: tf.Tensor) -> tf.Tensor:
        return self.dense(input)


class AddPositionalEncoding(Layer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.layer = Lambda(self._add_positional_encoding)

    def call(self, input: tf.Tensor) -> tf.Tensor:
        return self.layer(input)

    def _add_positional_encoding(self, input: tf.Tensor) -> tf.Tensor:
        i_type = input.dtype
        batch_size, length, dim = tf.unstack(tf.shape(input))

        # NOTE: expend_dims(x, 0)とすると[x]から[1, shape(x)]となる
        #       これに加えてtile(expand_dims(x, 0), [nums, 1])とすると[nums, shape(x)]となり
        #       テンソルxを縦にnums回並べたテンソルとなる

        dim_counter = tf.range(dim) // 2 * 2  # 0, 0, 2, 2, 4, ...
        dim_matrix = tf.tile(tf.expand_dims(dim_counter, 0), [length, 1])
        dim_matrix = tf.pow(10000.0, tf.cast(dim_matrix / dim, i_type))

        phase = (
            tf.cast(tf.range(dim) % 2, i_type) % math.pi / 2
        )  # 0, pi/2, 0, pi/2, ...
        phase_matrix = tf.tile(tf.expand_dims(phase, 0), [length, 1])

        pos_counter = tf.range(length)
        pos_matrix = tf.cast(tf.tile(tf.expand_dims(pos_counter, 1), [1, dim]), i_type)

        positional_encoding = tf.sin(pos_matrix / dim_matrix + phase_matrix)
        positional_encoding = tf.tile(
            tf.expand_dims(positional_encoding, 0), [batch_size, 1, 1]
        )

        return input + positional_encoding


class Encoder(Layer):
    def __init__(
        self,
        hidden_dim: int,
        num_layer: int,
        num_head: int,
        ffn_dim: int,
        dropout_rate: float,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.num_head = num_head
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout_rate

        self.token_embedding = TokenEmbedding(hidden_dim)
        self.add_position_encoding = AddPositionalEncoding()
        self.dropout = Dropout(dropout_rate)

        self.enc_blocks = []
        for _ in range(num_layer):
            attention_layer = SelfAttention(hidden_dim, num_head, dropout_rate)
            ffn_layer = FeedForwardNetwork(hidden_dim, ffn_dim, dropout_rate)
            self.enc_blocks.append(
                [
                    ResidualNormalizationWrapper(attention_layer, dropout_rate),
                    ResidualNormalizationWrapper(ffn_layer, dropout_rate),
                ]
            )

        self.layer_normalization = LayerNormalization()

    def call(self, input: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        embedded_input = self.token_embedding(input)
        embedded_input = self.add_position_encoding(embedded_input)
        query = self.dropout(embedded_input)

        for i, layers in enumerate(self.enc_blocks):
            attention_layer, ffn_layer = tuple(layers)

            query = attention_layer(query, mask=mask)
            query = ffn_layer(query)

        return self.layer_normalization(query)


class FeedForwardNetwork(Layer):
    def __init__(
        self, hidden_dim: int, ffn_dim: int, dropout_rate: float, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout_rate

        self.input_dense = Dense(ffn_dim, activation="relu")
        self.dropout = Dropout(dropout_rate)
        self.output_dense = Dense(hidden_dim)

    def call(self, input: tf.Tensor) -> tf.Tensor:
        output = self.input_dense(input)
        output = self.dropout(output)
        return self.output_dense(output)


class ResidualNormalizationWrapper(Layer):
    def __init__(self, layer: Layer, dropout_rate: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer = layer
        self.layer_normalization = LayerNormalization()
        self.dropout = Dropout(dropout_rate)
        self.add = Add()

    def call(self, input: tf.Tensor, **kwargs) -> tf.Tensor:
        output = self.layer_normalization(input)
        output = self.layer(output, **kwargs)
        output = self.dropout(output)

        return self.add([input, output])
