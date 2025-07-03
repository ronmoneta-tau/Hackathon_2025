import pytest
from src.metric_unit import MetricUnit

def test_metricunit_known_and_unknown():
    assert MetricUnit.get("SSRT") == "ms"
    assert MetricUnit.get("Post-Error Efficiency") == "%"
    assert MetricUnit.get("FooBar") == ""
