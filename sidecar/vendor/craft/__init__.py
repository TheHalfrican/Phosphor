# Vendored from clovaai/CRAFT-pytorch (MIT, (c) 2019-present NAVER Corp).
# See LICENSE in this directory. Only the model definition is vendored; we use the raw
# character-region heatmap as a mask and skip CRAFT's box/polygon postprocessing entirely.
from .craft import CRAFT
__all__ = ["CRAFT"]
