import re
from enum import Enum


class MetricUnit(Enum):
    """
    Enum mapping known metric names to their display units.
    """

    SSRT = "ms"
    Post_Go_Error_Efficiency = "%"
    D_context = "z-score"
    A_cue_bias = "z-score"
    PBI_error = "ratio"
    PBI_rt = "ratio"
    PBI_composite = "z-score"
    AX_CorrectRate = "%"
    AY_IncorrectRate = "%"

    @classmethod
    def get(cls, metric_name: str) -> str:
        """
        Return the unit string for a given metric.

        Parameters
        ----------
        metric_name : str
            The name of the metric, e.g. "SSRT" or "D’context".

        Returns
        -------
        str
            The unit for that metric, or an empty string if unknown.
        """
        # Normalize: lowercase, replace all types of apostrophes with underscore, remove punctuation
        key = metric_name
        # unify all apostrophes to underscore
        key = re.sub(r"[’'`]", "_", key)
        # remove dots and other punctuation except underscores and letters/numbers
        key = re.sub(r"[^\w]", "_", key)
        # collapse multiple underscores to one
        key = re.sub(r"_+", "_", key)
        # remove leading/trailing underscores/whitespace
        key = key.strip("_ ").strip()
        # upper/lower case fix to match Enum keys
        for candidate in cls.__members__:
            if candidate.lower() == key.lower():
                return cls[candidate].value
        return ""
