import numpy as np


class BaseEnhancer:
    def __init__(self, params: dict):
        self.params = params

    def get_params(self) -> dict:
        return self.params

    def get_name(self):
        if self.params.get('name') is None:
            raise ValueError("The name of enhancer is not defined. Please give a name of enhancer")

        return self.params.get('name')

    def enhance_image(self, image: np.array, *args, **kwargs) -> np.array:
        raise NotImplementedError("Method enhance_image is not implemented")
