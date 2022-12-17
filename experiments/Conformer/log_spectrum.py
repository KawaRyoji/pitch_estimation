import itertools

from deep_learning.dataset import DatasetParams
from deep_learning.experiment import DNNExperiment
from pitch_estimation.experiments.prepare_dataset import log_spectrum_2d
from pitch_estimation.models.Conformer import Conformer
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.metrics import F1Score


frame_lens = [1024, 2048]
normalize = [True]
frame_shift = 256
frame_num = 128
fft_point = 2048
train_method = "kcv"
dataset_params = DatasetParams(
    batch_size=32,
    epochs=100,
    batches_per_epoch=500,
)
k = 5
valid_split = 0.8

conformer = Conformer(
    Conformer.Params.medium(input_size=(frame_num, 1024)),
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
    train_set, test_set = log_spectrum_2d(
        dir="./resources/datasets",
        frame_len=frame_len,
        frame_shift=frame_shift,
        frame_num=frame_num,
        normalize=norm,
        fft_point=fft_point,
    )

    root_dir = "./results/Conformer/log_spectrum/l{}_s{}_t{}_n{}".format(
        frame_len, frame_shift, frame_num, norm
    )

    experiment = DNNExperiment(
        dnn=conformer,
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
