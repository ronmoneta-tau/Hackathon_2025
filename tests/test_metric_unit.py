from src.metric_unit import MetricUnit


def test_metricunit_get_canonical_names():
    assert MetricUnit.get("SSRT") == "ms"
    assert MetricUnit.get("Post_Go_Error_Efficiency") == "%"
    assert MetricUnit.get("D_context") == "z-score"
    assert MetricUnit.get("A_cue_bias") == "z-score"
    assert MetricUnit.get("PBI_error") == "ratio"
    assert MetricUnit.get("PBI_rt") == "ratio"
    assert MetricUnit.get("PBI_composite") == "z-score"
    assert MetricUnit.get("AX_CorrectRate") == "%"
    assert MetricUnit.get("AY_IncorrectRate") == "%"


def test_metricunit_get_names_with_spaces_or_dashes():
    assert MetricUnit.get("Post Go Error Efficiency") == "%"
    assert MetricUnit.get("Post-Go-Error-Efficiency") == "%"
    assert MetricUnit.get("D context") == "z-score"
    assert MetricUnit.get("A-cue bias") == "z-score"
    assert MetricUnit.get("AX CorrectRate") == "%"


def test_metricunit_get_names_with_apostrophes():
    assert MetricUnit.get("D'context") == "z-score"


def test_metricunit_get_names_with_dots_and_trim():
    assert MetricUnit.get("   Post Go Error Efficiency ") == "%"


def test_metricunit_get_case_sensitivity():
    assert MetricUnit.get("ssrt") == "ms"
    assert MetricUnit.get("SsRt") == "ms"
    assert MetricUnit.get("AX_Correctrate") == "%"


def test_metricunit_get_unknown_returns_empty():
    assert MetricUnit.get("Unknown") == ""
    assert MetricUnit.get("X") == ""
    assert MetricUnit.get("") == ""
