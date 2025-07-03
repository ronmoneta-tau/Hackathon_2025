from enum import Enum


class MetricUnit(Enum):
    """
    Enum mapping known metric names to their display units.
    """

    SSRT = "ms"
    Post_Error_Efficiency = "%"
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
        key = metric_name.replace(" ", "_").replace("’", "").replace("'", "")
        return cls[key].value if key in cls.__members__ else ""
