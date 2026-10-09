# Claims Data Quality and Exploratory Analysis Report

**Project:** Claim Intelligence Engine  
**Stage:** Task 02 — Data Validation, Cleaning, and Exploratory Data Analysis  
**Dataset:** Synthetic vehicle insurance claims (`data/raw/claims.csv`)  
**Records evaluated:** 1,200  
**Target variable:** `claim_amount`

## 1. Executive summary

The synthetic claims dataset passed the implemented data-quality checks without requiring row removal. All **1,200 records** and **16 columns** were retained in the processed dataset. No missing values, duplicate claim IDs, invalid policy dates, negative repair estimates, underage drivers, nonpositive vehicle values, or unsupported damage-severity categories were reported.

The claim amount distribution is **right-skewed**: the mean (**$7,411.91**) exceeds the median (**$5,418.97**). The interquartile-range (IQR) method flagged **71 potential high-value outliers (5.9% of records)**. These claims were retained because statistical extremity alone does not establish a data-quality error.

EDA also showed higher average claim amounts for more severe damage and a strong positive *visual* association between repair estimates and claim amounts. These patterns are consistent with the rules used to generate the synthetic target and should not be interpreted as validated real-world insurance relationships.

## 2. Dataset and validation results

| Check | Result |
|---|---:|
| Raw rows | 1,200 |
| Raw columns | 16 |
| Processed rows | 1,200 |
| Missing values | 0 |
| Duplicate claim IDs | 0 |
| Invalid policy dates | 0 |
| Negative repair estimates | 0 |
| Drivers younger than 18 | 0 |
| Nonpositive vehicle values | 0 |
| Invalid damage-severity categories | 0 |
| Records removed | 0 |

The cleaning module standardizes categorical text, converts numeric fields, parses date fields, and blocks export if any implemented validation check fails. The resulting dataset was saved to `data/processed/claims_processed.csv`.

These checks are appropriate for the current educational dataset, but they do not constitute a comprehensive production insurance-claims validation policy.

## 3. Exploratory findings

### 3.1 Claim amount distribution

| Statistic | Claim amount (USD) |
|---|---:|
| Count | 1,200 |
| Mean | $7,411.91 |
| Standard deviation | $6,195.97 |
| Minimum | $250.00 |
| 25th percentile (Q1) | $3,061.75 |
| Median | $5,418.97 |
| 75th percentile (Q3) | $9,918.02 |
| Maximum | $36,915.12 |

The histogram shows a concentration of lower-value claims and a longer tail of relatively expensive claims. The mean being above the median is consistent with this right-skewed shape.

**Figure:** [Claim amount distribution](figures/claim_distribution.png)

### 3.2 IQR outlier assessment

- **Lower IQR fence:** −$7,222.66
- **Upper IQR fence:** $20,202.43
- **Potential outliers:** 71 of 1,200 records (**5.9%**)

The IQR fences are statistical screening thresholds, not business validity limits. High-value claims were **not automatically removed or capped**. They will remain available for downstream modeling, where error on expensive claims should also be evaluated.

### 3.3 Average claim amount by damage severity

| Severity | Average claim (USD) |
|---|---:|
| Minor | $2,929.16 |
| Moderate | $7,566.48 |
| Severe | $16,922.13 |

Average claim amounts increase substantially with severity. Because severity contributed directly to the synthetic data-generation formula, the relationship is expected rather than evidence of independent causal discovery.

**Figure:** [Average claim by severity](figures/average_claim_by_severity.png)

### 3.4 Incident type distribution

| Incident type | Claims |
|---|---:|
| Collision | 631 |
| Glass | 199 |
| Hail | 162 |
| Vandalism | 115 |
| Theft | 93 |
| **Total** | **1,200** |

Collision is the most frequent incident category. This imbalance should be considered when comparing model performance across incident types.

**Figure:** [Incident type frequency](figures/Incident_frequency.png)

### 3.5 Repair estimate vs. claim amount

The scatter plot shows a **strong positive visual relationship** between repair estimates and claim amounts. No numerical correlation coefficient was calculated in this task, so no numerical correlation strength is claimed.

This relationship is also expected: the synthetic target-generation formula directly uses `repair_estimate`. Whether a repair estimate is appropriate as a model feature depends on its availability at prediction time; our planned app explicitly asks the user for an estimate before prediction.

**Figure:** [Repair estimate vs. claim amount](figures/repair_vs_claim.png)

## 4. Data decisions and limitations

1. **Retain all 1,200 rows.** No implemented validity check failed; no cleaning-based exclusions were needed.
2. **Retain the 71 IQR-flagged claims.** These are potential statistical outliers, not confirmed errors. Revisit their effect during model evaluation.
3. **Avoid target leakage.** `claim_amount` is the prediction target and must not be included in model features; `claim_id` is a record identifier and should also be excluded.
4. **Keep train-time and inference-time features consistent.** Feature generation and preprocessing must be reusable when a user submits a new claim.
5. **Treat this dataset as synthetic.** Patterns and future model scores demonstrate implementation and benchmarking, not real-world insurance pricing or settlement accuracy.

## 5. Terminal execution evidence

### `python -m src.clean_data`

```text
Raw dataset
Rows: 1200

Data quality report
missing_values: 0
duplicate_claim_ids: 0
invalid_policy_dates: 0
negative_repair_estimates: 0
invalid_driver_age: 0
invalid_vehicle_value: 0
invalid_damage_severity: 0

Cleaning complete
Rows remaining: 1200
Saved to: /Volumes/SSD/Projects/applied-agentic-ai-projects/05-claim-intelligence-engine/data/processed/claims_processed.csv
```

### `python -m src.explore_data`

```text
Claim amount statistics
count     1200.000000
mean      7411.906742
std       6195.974772
min        250.000000
25%       3061.745000
50%       5418.970000
75%       9918.017500
max      36915.120000
Name: claim_amount, dtype: float64

Outlier analysis
Lower bound: -7222.66
Upper bound: 20202.43
Potential outliers: 71

EDA complete
```

## 6. Conclusion and next step

Task 02 established a reproducible cleaning-and-EDA workflow, preserved the original 1,200 valid synthetic records, and identified the distributional characteristics that should inform modeling decisions.

**Next:** Task 03 — SQLite integration and historical claim comparisons. Later, evaluate a baseline prediction model against an engineered-feature model using a held-out test set, including attention to higher-value claims.
