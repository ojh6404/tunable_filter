#!/usr/bin/env python3
from abc import ABC, abstractmethod
from copy import deepcopy
import argparse
from dataclasses import dataclass
import cv2
import numpy as np
from typing import Dict, List, Optional

_window_name = 'window'
_initialized = {'?': False}

@dataclass
class TrackBarConfig:
    name: str
    val_min: int
    val_max: int


class Tunable(ABC):

    @abstractmethod
    def reflect_trackbar(self) -> None:
        pass

    @abstractmethod
    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        pass

    def start_tuning(self, img: np.ndarray):
        assert img.ndim == 3
        assert img.dtype == np.uint8
        while True:
            img_show = self.__call__(img)
            cv2.imshow(_window_name, img_show)
            self.reflect_trackbar()
            if cv2.waitKey(50) == ord('q'):
                break


@dataclass
class TunablePrimitive(Tunable):
    configs: List[TrackBarConfig]
    values: Optional[Dict[str, int]] = None
    window_name: str = _window_name

    def __post_init__(self):
        if not _initialized['?']:
            cv2.namedWindow(_window_name)
            _initialized['?'] = True
            print('initialize window')

        if self.values is None:
            # auto set initial values
            self.values = {}
            for config in self.configs:
                cv2.createTrackbar(
                        config.name,
                        self.window_name,
                        config.val_min,
                        config.val_max,
                        lambda x: None)
                self.values[config.name] = int(0.5 * (config.val_min + config.val_max))

    def reflect_trackbar(self):
        for config in self.configs:
            self.values[config.name] = cv2.getTrackbarPos(config.name, self.window_name)


class LogicalFilterBase(TunablePrimitive):

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        assert rgb.ndim == 3
        assert rgb.dtype == np.uint8
        out = self._call_impl(rgb)
        assert out.ndim == 2
        assert out.dtype == bool
        return out

    @abstractmethod
    def _call_impl(self, rgb: np.ndarray) -> np.ndarray:
        pass


class FilterBase(TunablePrimitive):

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        assert rgb.ndim == 3
        assert rgb.dtype == np.uint8
        out = self._call_impl(rgb)
        assert out.ndim == 3
        assert out.dtype == np.uint8
        return out

    @abstractmethod
    def _call_impl(self, rgb: np.ndarray) -> np.ndarray:
        pass


class CropLogicalFilter(LogicalFilterBase):

    @classmethod
    def from_image(cls, rgb: np.ndarray):
        width, height, _ = rgb.shape
        configs = []
        configs.append(TrackBarConfig('crop_x_min', 0, width))
        configs.append(TrackBarConfig('crop_x_max', 0, width))
        configs.append(TrackBarConfig('crop_y_min', 0, height))
        configs.append(TrackBarConfig('crop_y_max', 0, height))
        return cls(configs)

    def _call_impl(self, rgb: np.ndarray) -> np.ndarray:
        assert self.values is not None
        arr = np.ones(rgb.shape[:2], dtype=bool)
        arr[:self.values['crop_x_min'], :] = False
        arr[self.values['crop_x_max']:, :] = False
        arr[:, :self.values['crop_y_min']] = False
        arr[:, self.values['crop_y_max']:] = False
        return arr


class HSVLogicalFilter(LogicalFilterBase):

    @classmethod
    def default(cls):
        configs = []
        configs.append(TrackBarConfig('h_min', 0, 255))
        configs.append(TrackBarConfig('h_max', 0, 255))
        configs.append(TrackBarConfig('s_min', 0, 255))
        configs.append(TrackBarConfig('s_max', 0, 255))
        configs.append(TrackBarConfig('v_min', 0, 255))
        configs.append(TrackBarConfig('v_max', 0, 255))
        return cls(configs)

    def _call_impl(self, rgb: np.ndarray) -> np.ndarray:
        assert self.values is not None
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)

        bool_mat = np.ones(rgb.shape[:2], dtype=bool)
        for i, t in enumerate(['h', 's', 'v']) :
            key_min = t + '_min'
            key_max = t + '_max'
            b_min = self.values[key_min]
            b_max = self.values[key_max]
            bool_mat_local = np.logical_and(hsv[:, :, i] >= b_min, hsv[:, :, i] <= b_max)
            bool_mat *= bool_mat_local
        return bool_mat


class GaussianBlurFilter(FilterBase):

    @classmethod
    def default(cls):
        configs = []
        configs.append(TrackBarConfig('kernel_width', 1, 20))
        return cls(configs)

    def _call_impl(self, rgb: np.ndarray) -> np.ndarray:
        assert self.values is not None
        width = self.values['kernel_width']
        blured = cv2.blur(rgb, (width, width))
        return blured


@dataclass
class CompositeFilter(Tunable):
    converters: List[FilterBase]
    segmetors: List[LogicalFilterBase]

    def __call__(self, img_inp: np.ndarray) -> np.ndarray:
        img_out = deepcopy(img_inp)
        for converter in self.converters:
            img_out = converter(img_out)
        bool_mat = np.ones(img_out.shape[:2], dtype=bool)
        for segmentor in self.segmetors:
            bool_mat *= segmentor(img_out)
        img_out[np.logical_not(bool_mat)] = (0, 0, 0)
        return img_out

    def reflect_trackbar(self) -> None:
        for primitive in self.converters + self.segmetors:
            primitive.reflect_trackbar()
