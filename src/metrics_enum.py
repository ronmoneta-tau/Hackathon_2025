from enum import Enum


class Metrics(Enum):
    Linear = "linear"
    Nearest = "nearest"
    Nearest_up = "nearest-up"
    Zero = ("zero",)
    Slinear = "slinear"
    Quadratic = ("quadratic",)
    Cubic = ("cubic",)
    Previuos = ("previous",)
    Next = "next"
