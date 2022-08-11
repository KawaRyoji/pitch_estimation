import os
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple
from deep_learning.util import dir2paths
from deep_learning.dataset import Dataset, DatasetConstructor
import numpy as np

from audio_processing.audio import (
    FrameParameter,
    FrameSeries,
    SpectrumParameter,
    WavFile,
)

original_bits = 32
original_fs = 44100

# ソロ楽器の楽曲
solo_instrumental_train = [
    2186,
    2241,
    2242,
    2243,
    2244,
    2288,
    2289,
    2659,
    2217,
    2218,
    2219,
    2220,
    2221,
    2222,
    2293,
    2294,
    2295,
    2296,
    2297,
    2202,
    2203,
    2204,
]
solo_instrumental_test = [2191, 2298]


class MusicNet:
    def __init__(self, path: str) -> None:
        self.path = path

    def train_paths(self) -> Tuple[List[str], List[str]]:
        data_paths = dir2paths(os.path.join(self.path, "train_data"))
        label_paths = dir2paths(os.path.join(self.path, "train_labels"))
        data_paths = sorted(data_paths)
        label_paths = sorted(label_paths)

        return data_paths, label_paths

    def test_paths(self) -> Tuple[List[str], List[str]]:
        data_paths = dir2paths(os.path.join(self.path, "test_data"))
        label_paths = dir2paths(os.path.join(self.path, "test_labels"))
        data_paths = sorted(data_paths)
        label_paths = sorted(label_paths)

        return data_paths, label_paths

    def solo_train_paths(self) -> Tuple[List[str], List[str]]:
        data_paths = list(
            map(
                lambda x: os.path.join(self.path, "train_data/{}.wav".format(x)),
                solo_instrumental_train,
            )
        )
        label_paths = list(
            map(
                lambda x: os.path.join(self.path, "train_labels/{}.csv".format(x)),
                solo_instrumental_train,
            )
        )

        return data_paths, label_paths

    def solo_test_paths(self) -> Tuple[List[str], List[str]]:
        data_paths = list(
            map(
                lambda x: os.path.join(self.path, "test_data/{}.wav".format(x)),
                solo_instrumental_test,
            )
        )
        label_paths = list(
            map(
                lambda x: os.path.join(self.path, "test_labels/{}.csv".format(x)),
                solo_instrumental_test,
            )
        )

        return data_paths, label_paths

    def to_dataset(
        self,
        frame_param: FrameParameter,
        feature: str = "waveform",
        normalize: bool = False,
        include_nyquist: bool = True,
        train_set_path: Optional[str] = None,
        test_set_path: Optional[str] = None,
        spectrum_param: Optional[SpectrumParameter] = None,
        time_len: Optional[int] = None,
        mel_bins: Optional[int] = None,
    ) -> Tuple[Dataset, Dataset]:
        data_paths, label_paths = self.train_paths()
        train_set_constructor = DatasetConstructor(
            data_paths, label_paths, self.construct_process
        )
        train_set = train_set_constructor.construct(
            frame_param,
            normalize=normalize,
            feature=feature,
            time_len=time_len,
            include_nyquist=include_nyquist,
            spectrum_param=spectrum_param,
            mel_bins=mel_bins,
        )

        if train_set_path is not None:
            train_set.save(train_set_path)

        data_paths, label_paths = self.test_paths()
        test_set_constructor = DatasetConstructor(
            data_paths, label_paths, self.construct_process
        )
        test_set = test_set_constructor.construct(
            frame_param,
            normalize=normalize,
            feature=feature,
            time_len=time_len,
            include_nyquist=include_nyquist,
            spectrum_param=spectrum_param,
            mel_bins=mel_bins,
        )

        if test_set_path is not None:
            test_set.save(test_set_path)

        return train_set, test_set

    @staticmethod
    def construct_process(
        data_path: str,
        label_path: str,
        frame_param: FrameParameter,
        feature="waveform",
        time_len: Optional[int] = None,
        include_nyquist: bool = True,
        spectrum_param: Optional[SpectrumParameter] = None,
        mel_bins: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        wav_file = WavFile.read(data_path)
        data = wav_file.data
        label = Label.load(label_path)

        series = FrameSeries.from_param(data, frame_param)

        data = FrameSeries.feature_series(
            feature_name=feature,
            series=series,
            include_nyquist=include_nyquist,
            spectrum_param=spectrum_param,
            fs=wav_file.fs,
            mel_bins=mel_bins,
        )
        frame_label = label.to_frame_label(series.num_frame, frame_param)

        if time_len is not None:
            data_np = data.to_image(time_len)
            frame_label = FrameSeries(frame_label).to_image(time_len)
        else:
            data_np = data.frame_series

        return data_np, frame_label


class LabelInfo(Enum):
    START_TIME = "start_time"
    END_TIME = "end_time"
    INSTRUMENT = "instrument"
    NOTE = "note"
    START_BEAT = "start_beat"
    END_BEAT = "end_beat"
    NOTE_VALUE = "note_value"

    @staticmethod
    def header() -> str:
        return "start_time,end_time,instrument,note,start_beat,end_beat,note_value"

    @staticmethod
    def num_notes() -> int:
        return 128


class Label:
    def __init__(
        self,
        start_time: np.ndarray,
        end_time: np.ndarray,
        instrument: np.ndarray,
        note: np.ndarray,
        start_beat: np.ndarray,
        end_beat: np.ndarray,
        note_value: np.ndarray,
    ) -> None:
        self.start_time = start_time
        self.end_time = end_time
        self.instrument = instrument
        self.note = note
        self.start_beat = start_beat
        self.end_beat = end_beat
        self.note_value = note_value

    @classmethod
    def from_structured_array(cls, array: np.ndarray):
        return cls(
            array[LabelInfo.START_TIME.value],
            array[LabelInfo.END_TIME.value],
            array[LabelInfo.INSTRUMENT.value],
            array[LabelInfo.NOTE.value],
            array[LabelInfo.START_BEAT.value],
            array[LabelInfo.END_BEAT.value],
            array[LabelInfo.NOTE_VALUE.value],
        )

    def to_structured_array(self):
        data = np.zeros(
            len(self.start_time),
            dtype=[
                (LabelInfo.START_TIME.value, "<i4"),
                (LabelInfo.END_TIME.value, "<i4"),
                (LabelInfo.INSTRUMENT.value, "<i4"),
                (LabelInfo.NOTE.value, "<i4"),
                (LabelInfo.START_BEAT.value, "<f8"),
                (LabelInfo.END_BEAT.value, "<f8"),
                (LabelInfo.NOTE_VALUE.value, "<U26"),
            ],
        )

        data[LabelInfo.START_TIME.value] = self.start_time
        data[LabelInfo.END_TIME.value] = self.end_time
        data[LabelInfo.INSTRUMENT.value] = self.instrument
        data[LabelInfo.NOTE.value] = self.note
        data[LabelInfo.START_BEAT.value] = self.start_beat
        data[LabelInfo.END_BEAT.value] = self.end_beat
        data[LabelInfo.NOTE_VALUE.value] = self.note_value

        return data

    def convert_fs(self, origin_fs: int, target_fs: int) -> "Label":
        if origin_fs == target_fs:
            return self
        elif origin_fs < target_fs:
            raise RuntimeError("origin_fs > target_fs")

        fs_ratio = target_fs / origin_fs

        start_time = np.ceil(self.start_time * fs_ratio)
        end_time = np.ceil(self.end_time * fs_ratio)

        return Label(
            start_time=start_time,
            end_time=end_time,
            instrument=self.instrument,
            note=self.note,
            start_beat=self.start_beat,
            end_beat=self.end_beat,
            note_value=self.note_value,
        )

    def to_frame_label(
        self, num_frame: int, frame_param: FrameParameter, dtype=np.float32
    ) -> np.ndarray:
        frame_label = []
        for i in range(num_frame):
            start, end = FrameSeries.edge_point(
                i, frame_param.frame_len, frame_param.frame_shift
            )

            frame_pitches = self.notes_in_frame(start, end)
            frame_label.append(Label.note2hot_vector(frame_pitches))

        return np.array(frame_label, dtype=dtype)

    def notes_in_frame(self, frame_start: int, frame_end: int) -> np.ndarray:
        mid = (frame_start + frame_end) // 2

        if mid < 0:
            raise IndexError()

        index = (self.start_time <= mid) & (self.end_time >= mid)
        index = np.array(index)
        notes = self.note[index]
        return np.array(notes)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        label = self.to_structured_array()
        np.savetxt(path, label, delimiter=",", fmt="%s", header=LabelInfo.header())

    @classmethod
    def load(cls, path: str) -> "Label":
        label = np.genfromtxt(
            path, delimiter=",", names=True, dtype=None, encoding="utf-8"
        )
        return cls.from_structured_array(label)

    @staticmethod
    def note2hot_vector(label: np.ndarray):
        vector = np.zeros(LabelInfo.num_notes())
        vector[label] = 1
        vector = vector.astype(np.float32)

        return vector
