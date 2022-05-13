import numpy as np
import cv2 as cv
from .base import BaseEnhancer
from .utils import params_checker


class GlobalHistogramEqualization(BaseEnhancer):
    def __init__(self, params):
        super(GlobalHistogramEqualization, self).__init__(params)

    def enhance_image(self, image: np.array, *args, **kwargs) -> np.array:
        return np.array(list(map(cv.equalizeHist, image)))


class GammaCorrection(BaseEnhancer):
    def __init__(self, params):
        super(GammaCorrection, self).__init__(params)
        params_checker(('gamma',), self.params)

    def enhance_image(self, image: np.ndarray, gamma: np.float = 1.0, *args, **kwargs) -> np.array:
        gamma = gamma if gamma > 0 else 0.1
        inv_gamma = 1 / gamma

        table = [((i / 255) ** inv_gamma) * 255 for i in range(256)]
        table = np.array(table, np.uint8)

        return cv.LUT(image, table)


class LocalHistogramEqualization(BaseEnhancer):
    def __init__(self, params):
        super(LocalHistogramEqualization, self).__init__(params)
        params_checker(('clip_limit', 'tile_grid_size'), self.params)

    def enhance_image(self, image: np.ndarray, clip_limit: float = 2.5, tile_grid_size: int = 10):
        clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(int(tile_grid_size), int(tile_grid_size)))
        lab = cv.cvtColor(image, cv.COLOR_RGB2LAB)
        lab_planes = list(cv.split(lab))
        lab_planes[0] = clahe.apply(lab_planes[0])

        lab = cv.merge(lab_planes)

        return cv.cvtColor(lab, cv.COLOR_LAB2RGB)
