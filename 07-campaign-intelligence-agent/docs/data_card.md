# Data card and leakage audit

## Source
- Dataset: UCI Machine Learning Repository, Bank Marketing (ID 222)
- DOI: https://doi.org/10.24432/C5K306
- Landing page: https://archive.ics.uci.edu/dataset/222/bank+marketing
- License: CC BY 4.0; preserve attribution
- Edition: `bank-full.csv`, 45,211 examples, 16 inputs + target `y`, ordered by date according to UCI
- Origin: historical Portuguese telephone campaigns; NOT current banking customers

## Prediction contract
Estimate term-deposit subscription **before** a new outreach contact, given
available customer and previous-campaign information.

Pre-contact features: age, balance, pdays, previous, job, marital,
education, default, housing, loan, poutcome.

Excluded: `duration` (after-call), `campaign` (includes current contact),
`contact`, `day`, `month` (recorded as latest contact data, not necessarily
known at planning time). `y` is never a feature. This is a conservative
operational assumption; verify actual availability before real deployment.

UCI uses `unknown` for some categorical values; this is a category, not a
proof of no missingness. `pdays=-1` means no recorded previous contact.

## Evaluation protocol
Source-order split: first 70% training, next 15% validation, final 15% holdout.
The dataset does not contain explicit customer IDs or complete date stamps,
so source-order splitting cannot guarantee against repeated-customer leakage
or reconstruct a rigorous real-world time split.

The held-out test set is not a tuning playground. Report class prevalence,
precision, recall, F1, PR-AUC, ROC-AUC, Brier score, and failures.
The saved 0.5 threshold is just a default; consider threshold tuning on
validation only, before a single final holdout evaluation.

## Responsible use
- Demonstration only: no production targeting, financial advice, or automated
  decision about individuals.
- Age and marital status are sensitive/proxy attributes with fairness concerns;
  review subgroup disparities before using recommendations.
- Historical Portuguese banking campaigns are not a representative sample of
  current campaigns elsewhere. Covariate shift and sampling bias are expected.
- An estimated probability is not a causal effect or a guaranteed outcome.
- No customer IDs, so all row IDs created by ingestion are synthetic sequence keys.
