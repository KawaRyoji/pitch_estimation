import itertools

from deep_learning.dataset import DatasetParams
from deep_learning.experiment import KCVExperiment
from pitch_estimation.experiments.prepare_dataset import log_spectrum_1d
from pitch_estimation.models.CREPE import CREPE
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.metrics import F1Score


crepe = CREPE(
    CREPE.Params(),
    "binary_crossentropy",
    Adam(learning_rate=0.0001),
    metrics=[
        Precision(name="precision"),
        Recall(name="recall"),
        F1Score(num_classes=128, threshold=0.5, average="micro", name="F1"),
        AUC(curve="PR"),
    ],
)

frame_lens = [512, 1024, 2048]
normalize = [True, False]
frame_shift = 256
fft_point = 2048
dataset_params = DatasetParams(
    batch_size=32,
    epochs=100,
    steps_per_epoch=500,
)
k = 5

for frame_len, norm in itertools.product(frame_lens, normalize):
    train_set, test_set = log_spectrum_1d(
        dir="./resources/datasets",
        frame_len=frame_len,
        frame_shift=frame_shift,
        normalize=norm,
        fft_point=fft_point,
    )

    root_dir = "./results/CREPE/log_spectrum/l{}_s{}_n{}".format(
        frame_len, frame_shift, norm
    )

    experiment = KCVExperiment(
        dnn=crepe,
        root_dir=root_dir,
        train_set=train_set,
        test_set=test_set,
        dataset_params=dataset_params,
        k=k,
        gpu=0,
    )

    experiment.train()
    experiment.test()
    experiment.plot()
