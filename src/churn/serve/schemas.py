"""Request/response contracts for the prediction API.

Pydantic rejects malformed input with a 422 before it ever reaches the model.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Customer(BaseModel):
    tenure_months: int = Field(ge=0, le=72, examples=[5])
    monthly_charges: float = Field(ge=0, le=500, examples=[89.5])
    support_calls: int = Field(ge=0, examples=[3])
    senior_citizen: Literal[0, 1] = Field(examples=[0])
    contract: Literal["month_to_month", "one_year", "two_year"]
    internet_service: Literal["dsl", "fiber", "none"]
    payment_method: Literal["electronic_check", "mailed_check", "bank_transfer", "credit_card"]


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    model_version: str


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[Prediction]
