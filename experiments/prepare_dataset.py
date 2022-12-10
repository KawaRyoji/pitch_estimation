import os
from audio_processing.audio import FrameParameter, SpectrumParameter
from deep_learning.dataset import Dataset
from pitch_estimation.musicnet import MusicNet

musicnet = MusicNet("./resources/musicnet16k")


def waveform_1d(
    dir: str,
    frame_len: int,
    frame_shift: int,
    normalize: bool,
):
    train_path = os.path.join(
        dir, "waveform/train_l{}_s{}_n{}.npz".format(frame_len, frame_shift, normalize)
    )
    test_path = os.path.join(
        dir, "waveform/test_l{}_s{}_n{}.npz".format(frame_len, frame_shift, normalize)
    )

    if os.path.exists(train_path) and os.path.exists(test_path):
        train_set = Dataset.load(train_path)
        test_set = Dataset.load(test_path)
    else:
        frame_param = FrameParameter(frame_len=frame_len, frame_shift=frame_shift)
        train_set, test_set = musicnet.to_dataset(
            frame_param,
            feature="waveform",
            normalize=normalize,
            train_set_path=train_path,
            test_set_path=test_path,
        )

    return train_set, test_set


def spectrum_1d(
    dir: str,
    frame_len: int,
    frame_shift: int,
    normalize: bool,
    fft_point: int,
):
    train_path = os.path.join(
        dir, "spectrum/train_l{}_s{}_n{}.npz".format(frame_len, frame_shift, normalize)
    )
    test_path = os.path.join(
        dir, "spectrum/test_l{}_s{}_n{}.npz".format(frame_len, frame_shift, normalize)
    )

    if os.path.exists(train_path) and os.path.exists(test_path):
        train_set = Dataset.load(train_path)
        test_set = Dataset.load(test_path)
    else:
        frame_param = FrameParameter(frame_len=frame_len, frame_shift=frame_shift)
        spectrum_param = SpectrumParameter(fft_point=fft_point, window="hann")
        train_set, test_set = musicnet.to_dataset(
            frame_param=frame_param,
            feature="spectrum",
            normalize=normalize,
            include_nyquist=False,
            train_set_path=train_path,
            test_set_path=test_path,
            spectrum_param=spectrum_param,
        )

    return train_set, test_set


def log_spectrum_1d(
    dir: str,
    frame_len: int,
    frame_shift: int,
    normalize: bool,
    fft_point: int,
):
    train_path = os.path.join(
        dir,
        "log_spectrum/train_l{}_s{}_n{}.npz".format(frame_len, frame_shift, normalize),
    )
    test_path = os.path.join(
        dir,
        "log_spectrum/test_l{}_s{}_n{}.npz".format(frame_len, frame_shift, normalize),
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


def waveform_2d(
    dir: str,
    frame_len: int,
    frame_shift: int,
    frame_num: int,
    normalize: bool,
):
    train_path = os.path.join(
        dir,
        "waveform2D/train_l{}_s{}_t{}_n{}.npz".format(
            frame_len, frame_shift, frame_num, normalize
        ),
    )
    test_path = os.path.join(
        dir,
        "waveform2D/test_l{}_s{}_t{}_n{}.npz".format(
            frame_len, frame_shift, frame_num, normalize
        ),
    )

    if os.path.exists(train_path) and os.path.exists(test_path):
        train_set = Dataset.load(train_path)
        test_set = Dataset.load(test_path)
    else:
        frame_param = FrameParameter(frame_len=frame_len, frame_shift=frame_shift)
        train_set, test_set = musicnet.to_dataset(
            frame_param,
            feature="waveform",
            time_len=frame_num,
            normalize=normalize,
            train_set_path=train_path,
            test_set_path=test_path,
        )

    return train_set, test_set


def spectrum_2d(
    dir: str,
    frame_len: int,
    frame_shift: int,
    frame_num: int,
    normalize: bool,
    fft_point: int,
):
    train_path = os.path.join(
        dir,
        "spectrum2D/train_l{}_s{}_t{}_n{}.npz".format(
            frame_len, frame_shift, frame_num, normalize
        ),
    )
    test_path = os.path.join(
        dir,
        "spectrum2D/test_l{}_s{}_t{}_n{}.npz".format(
            frame_len, frame_shift, frame_num, normalize
        ),
    )

    if os.path.exists(train_path) and os.path.exists(test_path):
        train_set = Dataset.load(train_path)
        test_set = Dataset.load(test_path)
    else:
        frame_param = FrameParameter(frame_len=frame_len, frame_shift=frame_shift)
        spectrum_param = SpectrumParameter(fft_point=fft_point, window="hann")
        train_set, test_set = musicnet.to_dataset(
            frame_param=frame_param,
            feature="spectrum",
            normalize=normalize,
            time_len=frame_num,
            include_nyquist=False,
            train_set_path=train_path,
            test_set_path=test_path,
            spectrum_param=spectrum_param,
        )

    return train_set, test_set


def log_spectrum_2d(
    dir: str,
    frame_len: int,
    frame_shift: int,
    frame_num: int,
    normalize: bool,
    fft_point: int,
):
    train_path = os.path.join(
        dir,
        "log_spectrum2D/train_l{}_s{}_t{}_n{}.npz".format(
            frame_len, frame_shift, frame_num, normalize
        ),
    )
    test_path = os.path.join(
        dir,
        "log_spectrum2D/test_l{}_s{}_t{}_n{}.npz".format(
            frame_len, frame_shift, frame_num, normalize
        ),
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
            time_len=frame_num,
            include_nyquist=False,
            train_set_path=train_path,
            test_set_path=test_path,
            spectrum_param=spectrum_param,
        )

    return train_set, test_set
