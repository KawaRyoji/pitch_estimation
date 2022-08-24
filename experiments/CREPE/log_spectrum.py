import itertools
import os

from deep_learning.dataset import Dataset, DatasetParams
from deep_learning.experiment import DNNExperiment
from audio_processing.audio import FrameParameter, SpectrumParameter
from pitch_estimation.models import CREPE
from pitch_estimation.musicnet import MusicNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from tensorflow_addons.metrics import F1Score

musicnet = MusicNet("./resources/musicnet16k")


def prepare_dataset(frame_len: int, frame_shift: int, normalize: bool, fft_point: int):
    train_path = "./resources/datasets/log_spectrum/train_l{}_s{}_n{}.npz".format(
        frame_len, frame_shift, normalize
    )
    test_path = "./resources/datasets/log_spectrum/test_l{}_s{}_n{}.npz".format(
        frame_len, frame_shift, normalize
    )

    if os.path.exists(train_path) and os.path.exists(test_path):
        train_set = Dataset.load(train_path)
        test_set = Dataset.load(test_path)
    else:
        frame_param = FrameParameter(frame_len=frame_len, frame_shift=frame_shift)
        spectrum_param = SpectrumParameter(fft_point=fft_point, window="hann")
        train_set, test_set = musicnet.to_dataset(
            frame_param=frame_param,
            feature="log spectrum",
            normalize=normalize,
            include_nyquist=False,
            train_set_path=train_path,
            test_set_path=test_path,
            spectrum_param=spectrum_param,
        )

    return train_set, test_set


crepe = CREPE(
    CREPE.Params(),
    "binary_crossentropy",
    Adam(learning_rate=0.0001),
    metrics=[
        Precision(name="precision"),
        Recall(name="recall"),
        F1Score(num_classes=128, threshold=0.5, average="micro", name="F1"),
    ],
)

frame_lens = [512, 1024, 2048]
normalize = [True, False]
frame_shift = 256
fft_point = 2048
train_method = "holdout"
dataset_params = DatasetParams(
    batch_size=32,
    epochs=100,
    batches_per_epoch=500,
)
k = 5
valid_split = 0.8

for frame_len, norm in itertools.product(frame_lens, normalize):
    train_set, test_set = prepare_dataset(
        frame_len=frame_len,
        frame_shift=frame_shift,
        normalize=norm,
        fft_point=fft_point,
    )

    root_dir = "./results/CREPE/log_spectrum/l{}_s{}_n{}".format(
        frame_len, frame_shift, norm
    )

    experiment = DNNExperiment(
        dnn=crepe,
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
