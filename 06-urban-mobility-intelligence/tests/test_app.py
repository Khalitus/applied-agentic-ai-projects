import app


def test_form_renders_model_and_matched_historical_reference(monkeypatch):
    captured = {}

    def estimate(**kwargs):
        captured.update(kwargs)
        return {
            "estimated_rentals": 471.7,
            "scenario": {"hour": 8, "workingday": 1, "weather_code": 1},
            "historical_reference": {
                "observations": 25,
                "avg_rentals": 150.0,
                "min_rentals": 10,
                "max_rentals": 350,
            },
            "reference_sample": "small_sample",
        }

    monkeypatch.setattr(app, "estimate_scenario", estimate)
    output = app.run_scenario("2012-08-15", 8.0, False, 1, 22, 60, 12)

    assert captured["hour"] == 8
    assert captured["weather_code"] == 1
    assert "471.7 rentals/hour" in output
    assert "25 observations" in output
    assert "150.00 rentals/hour" in output
    assert "Small sample" in output
    assert "not a live forecast" in output


def test_form_rejects_fractional_hour():
    output = app.run_scenario("2012-08-15", 8.5, False, 1, 22, 60, 12)
    assert "Hour must be an integer" in output


def test_form_does_not_present_empty_reference_as_zero():
    result = {
        "estimated_rentals": 40.0,
        "scenario": {"hour": 3, "workingday": 0, "weather_code": 4},
        "historical_reference": {
            "observations": 0,
            "avg_rentals": None,
            "min_rentals": None,
            "max_rentals": None,
        },
        "reference_sample": "no_matches",
    }
    output = app.format_prediction(result)
    assert "No historical observations matched" in output
    assert "No historical average can be reported" in output
