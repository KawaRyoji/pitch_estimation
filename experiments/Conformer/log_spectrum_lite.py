import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from tensorflow.keras.optimizers import Adam
from deep_learning.dataset import DataSequence, Dataset, DatasetParams
from pitch_estimation.models.Conformer import Conformer
from tensorflow.keras.metrics import Recall, Precision, AUC
from tensorflow_addons.metrics import F1Score

gpus = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)

params = DatasetParams(
    batch_size=32,
    epochs=100,
    steps_per_epoch=500,
)

data = np.load("./resources/datasets/log_spectrum2D/train_l1024_s256_t16_nTrue.npz")
x = data["x"]
y = data["y"]
del data

split_point = 0.8
train_size = int(x.shape[0] * split_point)

train_x = x[:train_size]
train_y = y[:train_size]
valid_x = x[train_size:]
valid_y = y[train_size:]
train_sequence = DataSequence(train_x, train_y, 32, steps_per_epoch=400, shuffle=True)
valid_sequence = DataSequence(valid_x, valid_y, 32, steps_per_epoch=100, shuffle=False)
        
model_param = Conformer.Params.small()
conformer = Conformer(
    param=model_param,
    loss="binary_crossentropy",
    optimizer=Adam(learning_rate=0.0001),
        metrics=[
        Precision(name="precision"),
        Recall(name="recall"),
        F1Score(num_classes=128, threshold=0.5, average="micro", name="F1"),
        AUC(curve="PR"),
    ],
)

conformer.compile()
model = conformer.get_model()
model.fit(train_sequence, validation_data=valid_sequence, callbacks=[
    
])