import logging

import gradio as gr
import pandas as pd

from src.assessment import assess_claim
from src.config import PROCESSED_DATA_DIR
from src.predict import validate_claim
from src.config import ROOT_DIR

SPLIT_PATH = ROOT_DIR / "data" / "splits" / "claim_splits.csv"

logger = logging.getLogger(__name__)

DATA_PATH = PROCESSED_DATA_DIR / "claims_processed.csv"

INPUT_FIELDS = [
    "incident_date",
    "policy_start_date",
    "driver_age",
    "vehicle_age",
    "annual_mileage",
    "vehicle_value",
    "repair_estimate",
    "prior_claims",
    "incident_type",
    "vehicle_type",
    "weather",
    "region",
    "police_reported",
    "damage_severity",
]

CATEGORICAL_FIELDS = [
    "incident_type",
    "vehicle_type",
    "weather",
    "region",
    "police_reported",
    "damage_severity",
]


def load_category_choices() -> dict:
    data = pd.read_csv(DATA_PATH)
    splits = pd.read_csv(SPLIT_PATH)

    training_ids = set(
        splits.loc[splits["split"] == "train", "claim_id"]
    )

    data = data.loc[data["claim_id"].isin(training_ids)]

    choices = {}

    for column in CATEGORICAL_FIELDS:
        values = (
            data[column]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .unique()
            .tolist()
        )

        choices[column] = sorted(values)

        if not choices[column]:
            raise ValueError(
                f"No category choices found for {column}."
            )

    return choices


CATEGORY_CHOICES = load_category_choices()

DEMO_CLAIM = {
    "incident_date": "2026-09-15",
    "policy_start_date": "2024-06-01",
    "driver_age": 34,
    "vehicle_age": 5,
    "annual_mileage": 12000,
    "vehicle_value": 28000,
    "repair_estimate": 7800,
    "prior_claims": 1,
    "incident_type": "collision",
    "vehicle_type": "sedan",
    "weather": "clear",
    "region": "north",
    "police_reported": "yes",
    "damage_severity": "moderate",
}

def build_claim(values: list | tuple) -> dict:
    if len(values) != len(INPUT_FIELDS):
        raise ValueError("Incorrect number of claim inputs.")

    claim = dict(zip(INPUT_FIELDS, values))

    validate_claim(claim)

    return claim

def format_prediction(prediction: dict) -> str:
    amount = prediction["predicted_amount"]
    model = prediction["selected_model"]

    return (
        "### Preliminary ML estimate\n\n"
        f"## ${amount:,.2f}\n\n"
        f"Model: `{model}`\n\n"
        "This is an automated estimate, not an "
        "approved payment or final settlement decision."
    )

def format_items(items: list[str]) -> str:
    if not items:
        return "_None identified from the available evidence._"

    return "\n".join(
        f"- {item}"
        for item in items
    )

def format_assessment(result: dict) -> str:
    assessment = result["assessment"]

    sections = [
        "### Assessment summary",
        assessment["summary"],

        "### Documentation gaps",
        format_items(assessment["documentation_gaps"]),

        "### Recommended next steps",
        format_items(assessment["recommended_next_steps"]),

        "### Limitations",
        format_items(assessment["limitations"]),
    ]

    citations = [
        f'{citation["source_type"]}: {citation["source_id"]}'
        for citation in assessment["citations"]
    ]
    sections.extend([
        "### Citations",
        format_items(citations),
    ])

    policy_coverage_note = result.get("policy_coverage_note")
    if policy_coverage_note is not None:
        sections.extend([
            "### Policy coverage note",
            policy_coverage_note,
        ])

    return "\n\n".join(sections)

CASE_COLUMNS = [
    "Claim ID",
    "Incident",
    "Severity",
    "Vehicle",
    "Repair estimate",
    "Historical settlement",
    "Repair difference",
]


def format_historical_cases(evidence: dict) -> pd.DataFrame:
    rows = []

    for case in evidence["historical_cases"]:
        rows.append({
            "Claim ID": case["claim_id"],
            "Incident": case["incident_type"],
            "Severity": case["damage_severity"],
            "Vehicle": case["vehicle_type"],
            "Repair estimate": round(float(case["repair_estimate"]), 2),
            "Historical settlement": round(float(case["historical_claim_amount"]), 2),
            "Repair difference": round(float(case["repair_difference"]), 2),
        })

    return pd.DataFrame(rows, columns=CASE_COLUMNS)


POLICY_COLUMNS = [
    "Source file",
    "Chunk ID",
    "Passage",
]

def format_policy_evidence(evidence: dict) -> pd.DataFrame:
    rows = []

    for policy in evidence["policies"]:
        rows.append({
            "Source file": policy["source_file"],
            "Chunk ID": policy["chunk_id"],
            "Passage": policy["document"],
        })

    return pd.DataFrame(rows, columns=POLICY_COLUMNS)

def format_image_analysis(image_result: dict) -> str:
    if not image_result["image_provided"]:
        return "No photograph was provided."

    return (
        "### Automated image observations\n\n"
        f"**Generated caption:** {image_result['caption']}\n\n"
        f"**Model:** `{image_result['model_id']}`\n\n"
        f"**Limitations:** {image_result['limitations']}"
    )

def empty_outputs(message: str) -> tuple:
    return (
        message,
        "",
        "",
        pd.DataFrame(columns=CASE_COLUMNS),
        pd.DataFrame(columns=POLICY_COLUMNS),
        "",
    )

def run_assessment(*values) -> tuple:
    if len(values) != len(INPUT_FIELDS) + 1:
        return empty_outputs("Invalid input count.")

    claim_values = values[:-1]
    image_path = values[-1]

    try:
        claim = build_claim(claim_values)

        result = assess_claim(
            claim,
            image_path=image_path,
        )

        if not isinstance(result, dict) or not isinstance(
            result.get("assessment"), dict
        ):
            raise ValueError(
                "Assessment response is missing the validated assessment."
            )

        # `assess_claim` exposes the model output under `ml_prediction`.
        prediction_text = format_prediction(result["ml_prediction"])
        assessment_text = format_assessment(result)

        historical_table = format_historical_cases(result["evidence"])  
        policy_table = format_policy_evidence(result["evidence"])  

        image_text = format_image_analysis(result["image_analysis"])

        return (
                "Assessment completed. Synthetic demonstration only.",
                prediction_text,
                assessment_text,
                historical_table,
                policy_table,
                image_text,
            )

    except (ValueError, TypeError, FileNotFoundError) as exc:
        return empty_outputs(
            f"**Assessment unavailable:** {exc}"
        )

    except Exception:
        logger.exception("Claim assessment failed.")
        return empty_outputs(
            "**Assessment unavailable.** "
            "The assessment service could not complete "
            "the request. Check the local terminal."
        )

def category_dropdown(field: str, label: str):
    choices = CATEGORY_CHOICES[field]
    default = DEMO_CLAIM[field]

    if default not in choices:
        raise ValueError(
            f"Demo value {default!r} is not a training category."
        )

    return gr.Dropdown(
        choices=choices,
        value=default,
        label=label,
        interactive=True,
    )

def build_app():
    with gr.Blocks(
        title="Claim Intelligence Engine",
        theme=gr.themes.Base(),
    ) as demo:

        gr.Markdown(
            "# Claim Intelligence Engine\n"
            "Multimodal claim assessment with ML prediction, "
            "historical evidence, and policy retrieval.\n\n"
            "**Synthetic demonstration only. "
            "Not a coverage or settlement decision.**"
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Incident information")

                incident_date = gr.Textbox(
                    label="Incident date (YYYY-MM-DD)",
                    value=DEMO_CLAIM["incident_date"],
                )

                policy_start_date = gr.Textbox(
                    label="Policy start date (YYYY-MM-DD)",
                    value=DEMO_CLAIM["policy_start_date"],
                )

                incident_type = category_dropdown(
                    "incident_type", "Incident type"
                )

                damage_severity = category_dropdown(
                    "damage_severity", "Reported damage severity"
                )

                weather = category_dropdown(
                    "weather", "Weather"
                )

                region = category_dropdown(
                    "region", "Region"
                )

                police_reported = category_dropdown(
                    "police_reported", "Police reported"
                )

            with gr.Column():
                gr.Markdown("### Vehicle and driver")

                vehicle_type = category_dropdown(
                    "vehicle_type", "Vehicle type"
                )

                driver_age = gr.Number(
                    label="Driver age",
                    value=DEMO_CLAIM["driver_age"],
                    precision=0,
                )

                vehicle_age = gr.Number(
                    label="Vehicle age (years)",
                    value=DEMO_CLAIM["vehicle_age"],
                    precision=0,
                )

                annual_mileage = gr.Number(
                    label="Annual mileage",
                    value=DEMO_CLAIM["annual_mileage"],
                )

                vehicle_value = gr.Number(
                    label="Vehicle value ($)",
                    value=DEMO_CLAIM["vehicle_value"],
                )

                repair_estimate = gr.Number(
                    label="Repair estimate ($)",
                    value=DEMO_CLAIM["repair_estimate"],
                )

                prior_claims = gr.Number(
                    label="Previous claims",
                    value=DEMO_CLAIM["prior_claims"],
                    precision=0,
                )

        gr.Markdown("### Optional photographic evidence")

        image_input = gr.Image(
            label="Vehicle damage photograph",
            type="filepath",
            sources=["upload"],
            height=220,
        )

        submit = gr.Button(
            "Generate assessment",
            variant="primary",
        )

        status = gr.Markdown(
            "Enter or modify the synthetic claim details, "
            "then generate an assessment."
        )

        with gr.Tabs():
            with gr.Tab("Assessment"):
                prediction_output = gr.Markdown()
                assessment_output = gr.Markdown()
                image_output = gr.Markdown()

            with gr.Tab("Historical cases"):
                historical_output = gr.Dataframe(
                    label="Comparable synthetic claims",
                    headers=CASE_COLUMNS,
                    interactive=False,
                    wrap=True,
                )

            with gr.Tab("Policy evidence"):
                policy_output = gr.Dataframe(
                    label="Retrieved demonstration policies",
                    headers=POLICY_COLUMNS,
                    interactive=False,
                    wrap=True,
                )

        inputs = [
            incident_date,
            policy_start_date,
            driver_age,
            vehicle_age,
            annual_mileage,
            vehicle_value,
            repair_estimate,
            prior_claims,
            incident_type,
            vehicle_type,
            weather,
            region,
            police_reported,
            damage_severity,
            image_input,
        ]

        submit.click(
            fn=run_assessment,
            inputs=inputs,
            outputs=[
                status,
                prediction_output,
                assessment_output,
                historical_output,
                policy_output,
                image_output,
            ],
            concurrency_limit=1,
            show_progress="full",
        )

    demo.queue(
        max_size=8,
        default_concurrency_limit=1,
    )

    return demo

if __name__ == "__main__":
    app = build_app()

    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        max_file_size="10mb",
        enable_monitoring=False,
    )
