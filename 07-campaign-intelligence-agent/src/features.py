"""Pre-contact features only; see docs/data_card.md for leakage audit."""

NUMERIC = ["age", "balance", "pdays", "previous"]
CATEGORICAL = [
    "job", "marital", "education", "default", "housing", "loan", "poutcome"
]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "y"

# 'duration' is known only after a call, and 'campaign' includes current-call data.
# We exclude current-contact timing/channel fields as well.
EXCLUDED = ["duration", "campaign", "contact", "day", "month"]

JOB_CHOICES = [
    "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
    "retired", "self-employed", "services", "student", "technician",
    "unemployed", "unknown",
]
MARITAL_CHOICES = ["married", "single", "divorced"]
EDUCATION_CHOICES = ["primary", "secondary", "tertiary", "unknown"]
POUTCOME_CHOICES = ["unknown", "failure", "other", "success"]
