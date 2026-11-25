from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class CustomerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=18, le=100)
    balance: int = Field(ge=-100_000, le=1_000_000)
    pdays: int = Field(ge=-1, le=999)
    previous: int = Field(ge=0, le=500)
    job: Literal[
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed", "unknown",
    ]
    marital: Literal["married", "single", "divorced"]
    education: Literal["primary", "secondary", "tertiary", "unknown"]
    default: Literal["yes", "no"]
    housing: Literal["yes", "no"]
    loan: Literal["yes", "no"]
    poutcome: Literal["unknown", "failure", "other", "success"]

    @model_validator(mode="after")
    def verify_previous_history(self):
        if self.previous == 0 and self.poutcome != "unknown":
            raise ValueError("Previous outcome must be 'unknown' when previous=0")
        if self.previous == 0 and self.pdays != -1:
            raise ValueError("pdays must be -1 when previous=0")
        return self
