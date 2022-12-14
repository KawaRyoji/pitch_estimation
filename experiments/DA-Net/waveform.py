import itertools

from deep_learning.dataset import DatasetParams
from deep_learning.experiment import KCVExperiment
from pitch_estimation.experiments.prepare_dataset import waveform_1d
from pitch_estimation.models.DA_Net import DA_Net
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.metrics import F1Score


da_net = DA_Net(
    DA_Net.Params(),
    "binary_crossentropy",
    Adam(learning_rate=0.0001),
    metrics=[
        Precision(name="precision"),
        Recall(name="recall"),
        F1Score(num_classes=128, threshold=0.5, average="micro", name="F1"),
        AUC(curve="PR"),
    ],
)

frame_lens = [1024]
normalize = [True, False]
frame_shift = 256
train_method = "kcv"
dataset_params = DatasetParams(
    batch_size=32,
    epochs=100,
    steps_per_epoch=500,
)
k = 5
valid_split = 0.8

for frame_len, norm in itertools.product(frame_lens, normalize):
    train_set, test_set = waveform_1d(
        dir="./resources/datasets",
        frame_len=frame_len,
        frame_shift=frame_shift,
        normalize=norm,
    )

    root_dir = "./results/DA-Net/waveform/l{}_s{}_n{}".format(
        frame_len, frame_shift, norm
    )

    experiment = KCVExperiment(
        dnn=da_net,
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
