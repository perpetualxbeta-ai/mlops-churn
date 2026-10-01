"""Data contract for the raw customer table, enforced with Pandera.

Bad data is the most common way ML systems break. Validating at the pipeline
boundary turns silent model degradation into a loud, early failure.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from churn.config import CATEGORIES

customer_schema = pa.DataFrameSchema(
    {
        "customer_id": pa.Column(str, unique=True),
        "tenure_months": pa.Column(int, pa.Check.in_range(0, 72)),
        "monthly_charges": pa.Column(float, pa.Check.in_range(0, 500)),
        "support_calls": pa.Column(int, pa.Check.ge(0)),
        "senior_citizen": pa.Column(int, pa.Check.isin([0, 1])),
        "contract": pa.Column(str, pa.Check.isin(CATEGORIES["contract"])),
        "internet_service": pa.Column(str, pa.Check.isin(CATEGORIES["internet_service"])),
        "payment_method": pa.Column(str, pa.Check.isin(CATEGORIES["payment_method"])),
        "churn": pa.Column(int, pa.Check.isin([0, 1])),
    },
    strict=True,  # unexpected columns are an error, not something to ignore
    coerce=True,
)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Return the validated frame, or raise with a report of every failing check."""
    return customer_schema.validate(df, lazy=True)
