import numpy as np
import cv2 as cv
from .base import BaseEnhancer
from .utils import params_checker


class MedianFilterDenoiser(BaseEnhancer):
    def __init__(self, params):
        super(MedianFilterDenoiser, self).__init__(params)
        params_checker(('kernel_size',), self.params)

    def enhance_image(self, image: np.array, kernel_size: int = 9, *args, **kwargs) -> np.array:
        return cv.medianBlur(image, int(kernel_size))


class BilateralFilterDenoiser(BaseEnhancer):
    def __init__(self, params):
        super(BilateralFilterDenoiser, self).__init__(params)
        params_checker(('distance', 'sigma'), self.params)

    def enhance_image(self, image: np.array, distance: int = 10., sigma: int = 20, *args, **kwargs) -> np.array:
        return cv.bilateralFilter(image, int(distance), int(sigma), int(sigma))

