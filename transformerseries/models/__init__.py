from .transfomer import TransfomerSeries
from .lstm import LSTM
from .baseline import Baseline

ALL = {
    "transformer": TransformerSeries,
    "lstm": LSTM,
    "baseline": Baseline,
}