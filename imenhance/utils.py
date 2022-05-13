from typing import Tuple


def params_checker(params_to_check: Tuple[str, ...] = (), params: dict = {}):
    for param in params_to_check:
        if params.get(param) is None:
            raise ValueError(f"Need to have {param} parameter")
