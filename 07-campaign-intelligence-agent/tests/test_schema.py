import pytest
from pydantic import ValidationError
from src.schema import CustomerInput


@pytest.fixture
def customer():
    return {
        "age": 38, "balance": 2400, "pdays": -1, "previous": 0,
        "job": "technician", "marital": "married", "education": "secondary",
        "default": "no", "housing": "yes", "loan": "no", "poutcome": "unknown",
    }


def test_valid_input(customer):
    assert CustomerInput.model_validate(customer).age == 38


def test_rejects_inconsistent_history(customer):
    customer["poutcome"] = "success"
    with pytest.raises(ValidationError):
        CustomerInput.model_validate(customer)


def test_rejects_unrecognized_fields(customer):
    customer["duration"] = 250
    with pytest.raises(ValidationError):
        CustomerInput.model_validate(customer)
