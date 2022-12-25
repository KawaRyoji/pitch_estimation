import itertools

from deep_learning.dataset import DatasetParams
from deep_learning.experiment import KCVExperiment
from pitch_estimation.experiments.prepare_dataset import waveform_2d
from pitch_estimation.models.Transformer import Transformer
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.metrics import F1Score


frame_lens = [1024]
normalize = [True, False]
frame_shift = 256
frame_num = 16
train_method = "kcv"
dataset_params = DatasetParams(
    batch_size=32,
    epochs=100,
    steps_per_epoch=500,
)
k = 5
valid_split = 0.8

transformer = Transformer(
    Transformer.Params(data_length=frame_num),
    "binary_crossentropy",
    Adam(learning_rate=0.0001),
    metrics=[
        Precision(name="precision"),
        Recall(name="recall"),
        F1Score(num_classes=128, threshold=0.5, average="micro", name="F1"),
        AUC(curve="PR"),
    ],
)

for frame_len, norm in itertools.product(frame_lens, normalize):
    train_set, test_set = waveform_2d(
        dir="./resources/datasets",
        frame_len=frame_len,
        frame_shift=frame_shift,
        frame_num=frame_num,
        normalize=norm,
    )

    root_dir = "./results/Transformer/waveform/l{}_s{}_t{}_n{}".format(
        frame_len, frame_shift, frame_num, norm
    )

    experiment = KCVExperiment(
        dnn=transformer,
        root_dir=root_dir,
        train_set=train_set,
        test_set=test_set,
        dataset_params=dataset_params,
        train_method=train_method,
        k=k,
        valid_split=valid_split,
        gpu=0,
    )

    experiment.train()
    experiment.test()
    experiment.plot()
    del train_set
    del test_set
