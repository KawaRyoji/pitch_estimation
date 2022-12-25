import tensorflow as tf

def shape_list(x: tf.Tensor, dtype=tf.int32):
    static = x.shape.as_list()
    dynamic = tf.shape(x, out_type=dtype)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]