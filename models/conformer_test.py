import tensorflow as tf
from pitch_estimation.layers.embedding import PositionalEmbedding
from pitch_estimation.models.Conformer import (
    Conformer,
    ConvolutionModule,
    FeedForwardModule,
    MultiHeadedSelfAttentionModule,
    ConformerBlock,
)
from pitch_estimation.layers.common import RelativeMultiHeadAttention
from conformer_tf import ConformerConvModule

in_shape = (1, 128, 256)
x = tf.random.normal(in_shape)
print(ConvolutionModule(256)(x).shape)
print(FeedForwardModule(256)(x).shape)

rmha = RelativeMultiHeadAttention(dim=256, heads=4, dropout=0.1)

y = rmha(query=x, key=x, value=x, pos_emb=tf.zeros_like(x)).shape
print(y)

pe = PositionalEmbedding(256)(128)

print(pe.shape)

mhsa = MultiHeadedSelfAttentionModule(
    256,
    heads=4,
)

print(mhsa(x).shape)

block = ConformerBlock(256, heads=4)

print(block(x).shape)

conformer = Conformer(
    Conformer.Params.medium(input_size=(128, 256)), "binary_crossentropy"
)

conformer.definition(Conformer.Params.medium(input_size=(128, 256))).summary()
