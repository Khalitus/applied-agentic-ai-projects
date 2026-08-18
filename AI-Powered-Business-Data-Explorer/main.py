import os
import json
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

# # Load environment variables from the .env file in the same directory
# load_dotenv()

# # Get Gemini API key from environment variables
# gemini_api_key = os.getenv("GEMINI_API_KEY")

# # 1. Define desired data structure using Pydantic
# class TopicResearch(BaseModel):
#     topic: str = Field(description="The main subject of the research")
#     summary: str = Field(description="A brief two-sentence summary")
#     keywords: list[str] = Field(description="List of 3 related keywords")


# # 2. Initialize model and parser
# # Configure temperature, top_k, top_p, and the API key here
# model = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash",  # You can also use "gemini-1.5-pro" or "gemini-2.0-flash"
#     google_api_key=gemini_api_key,
#     temperature=0.7,  # Controls randomness (0.0 = deterministic, 1.0 = creative)
#     top_k=40,         # Selects from the top K most probable tokens
#     top_p=0.95,       # Nucleus sampling (cumulative probability cutoff)
# )

# parser = JsonOutputParser(pydantic_object=TopicResearch)

# # 3. Setup prompt with format instructions
# prompt = PromptTemplate(
#     template="Answer the user query.\n{format_instructions}\nQuery: {query}\n",
#     input_variables=["query"],
#     partial_variables={"format_instructions": parser.get_format_instructions()},
# )

# # 4. Create chain and invoke
# chain = prompt | model | parser

# result = chain.invoke({"query": "Artificial Intelligence in healthcare"})
# print(result)

# ============================================================
# PROJECT 1: AI-POWERED BUSINESS DATA EXPLORER
# ============================================================

CSV_FILE = "fake_retail_sales_messy.csv"
CLEANED_CSV_FILE = "cleaned_retail_sales.csv"
REPORT_FILE = "business_report.json"


# Columns that our project expects to find in the CSV
REQUIRED_COLUMNS = [
    "order_id",
    "order_date",
    "region",
    "category",
    "product",
    "customer",
    "quantity",
    "unit_price",
    "sales",
    "cost",
]


# ============================================================
# 1. STRUCTURE FOR THE AI REPORT
# ============================================================

class BusinessReport(BaseModel):
    executive_summary: str = Field(
        description="A short overall summary of business performance"
    )

    strongest_area: str = Field(
        description="The strongest business area based on the supplied data"
    )

    weakest_area: str = Field(
        description="The weakest business area based on the supplied data"
    )

    key_insights: list[str] = Field(
        description="Important insights supported by the supplied statistics"
    )

    recommendations: list[str] = Field(
        description="Practical management recommendations based on the analysis"
    )


# ============================================================
# 2. LOAD + BASICALLY INSPECT THE FULL DATASET
# ============================================================

def load_data(file_path):
    """Load the complete CSV and perform basic validation."""

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path}")

    # This loads the FULL CSV.
    # There is no nrows=5 and no slicing.
    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("The CSV file is empty.")

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    print("\nDATASET LOADED")
    print("-" * 40)
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    return df


# ============================================================
# 3. CLEAN THE DATA
# ============================================================

def clean_data(df):
    """Clean the main issues in the retail dataset."""

    # We make a copy so we do not accidentally change the original DataFrame
    df = df.copy()

    original_rows = len(df)

    # ----------------------------
    # Clean column names
    # ----------------------------

    df.columns = df.columns.str.strip().str.lower()


    # ----------------------------
    # Clean text columns
    # ----------------------------

    text_columns = [
        "region",
        "category",
        "product",
        "customer",
    ]

    for column in text_columns:
        df[column] = df[column].astype("string").str.strip()


    # Standardize capitalization
    # "karachi", "KARACHI", " Karachi "
    # all become "Karachi"

    df["region"] = df["region"].str.title()
    df["category"] = df["category"].str.title()
    df["product"] = df["product"].str.title()


    # Fill missing text values
    df["region"] = df["region"].fillna("Unknown")
    df["category"] = df["category"].fillna("Unknown")
    df["customer"] = df["customer"].fillna("Unknown")


    # ----------------------------
    # Clean numeric columns
    # ----------------------------

    # Some sales values look like:
    # 125,000.00
    #
    # Some cost values look like:
    # PKR 85,000.00
    #
    # Pandas may therefore treat the column as text.

    for column in ["sales", "cost"]:

        df[column] = (
            df[column]
            .astype("string")
            .str.replace("PKR", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


    # Convert other numeric columns
    df["quantity"] = pd.to_numeric(
        df["quantity"],
        errors="coerce"
    )

    df["unit_price"] = pd.to_numeric(
        df["unit_price"],
        errors="coerce"
    )


    # ----------------------------
    # Remove duplicate rows
    # ----------------------------

    df = df.drop_duplicates()


    # ----------------------------
    # Remove unusable financial rows
    # ----------------------------

    # If sales/cost/quantity are missing,
    # we cannot reliably calculate our business KPIs.

    df = df.dropna(
        subset=["sales", "cost", "quantity"]
    )


    # Basic human/business logic:
    # negative or zero sales/quantity do not make sense
    # for this simple sales dataset.

    df = df[
        (df["sales"] > 0)
        & (df["cost"] >= 0)
        & (df["quantity"] > 0)
    ]


    # ----------------------------
    # Feature engineering
    # ----------------------------

    df["profit"] = df["sales"] - df["cost"]

    df["profit_margin"] = np.where(
        df["sales"] != 0,
        (df["profit"] / df["sales"]) * 100,
        np.nan
    )


    print("\nDATA CLEANING COMPLETE")
    print("-" * 40)

    print(
        f"Rows removed during cleaning: "
        f"{original_rows - len(df)}"
    )

    print(f"Rows remaining: {len(df)}")

    return df


# ============================================================
# 4. BUSINESS ANALYSIS
# ============================================================

def analyze_data(df):
    """Calculate deterministic business metrics using Pandas."""

    total_sales = df["sales"].sum()
    total_cost = df["cost"].sum()
    total_profit = df["profit"].sum()

    total_orders = df["order_id"].nunique()

    average_order_value = (
        total_sales / total_orders
        if total_orders > 0
        else 0
    )

    overall_profit_margin = (
        (total_profit / total_sales) * 100
        if total_sales > 0
        else 0
    )


    # ----------------------------
    # Overall KPIs
    # ----------------------------

    kpis = {
        "total_orders": int(total_orders),

        "total_sales": round(
            float(total_sales), 2
        ),

        "total_cost": round(
            float(total_cost), 2
        ),

        "total_profit": round(
            float(total_profit), 2
        ),

        "average_order_value": round(
            float(average_order_value), 2
        ),

        "overall_profit_margin": round(
            float(overall_profit_margin), 2
        ),

        "total_quantity_sold": int(
            df["quantity"].sum()
        ),
    }


    # ----------------------------
    # Region analysis
    # ----------------------------

    # Study this carefully.
    #
    # We:
    # 1. group rows by region
    # 2. calculate several metrics
    # 3. sort highest sales first

    region_summary = (
        df
        .groupby("region", as_index=False)
        .agg(
            total_sales=("sales", "sum"),
            total_profit=("profit", "sum"),
            total_quantity=("quantity", "sum"),
            total_orders=("order_id", "nunique"),
        )
        .sort_values(
            "total_sales",
            ascending=False
        )
    )


    region_summary["profit_margin"] = np.where(
        region_summary["total_sales"] != 0,

        (
            region_summary["total_profit"]
            / region_summary["total_sales"]
        ) * 100,

        np.nan
    )


    # Round values so the AI receives cleaner numbers
    region_summary[
        ["total_sales", "total_profit", "profit_margin"]
    ] = region_summary[
        ["total_sales", "total_profit", "profit_margin"]
    ].round(2)


    # ========================================================
    # YOUR 30% — TODO 1
    # ========================================================
    #
    # Build category_summary yourself.
    #
    # It should work almost exactly like region_summary,
    # except group by:
    #
    #     "category"
    #
    # Calculate:
    #
    #     total_sales
    #     total_profit
    #     total_quantity
    #     total_orders
    #
    # Then sort by total_sales from highest to lowest.
    #
    # Hint:
    # Copy the region_summary pattern above and modify it.
    #
    # Replace this empty DataFrame when you complete it.

    category_summary = pd.DataFrame(
        columns=[
            "category",
            "total_sales",
            "total_profit",
            "total_quantity",
            "total_orders",
        ]
    )


    return kpis, region_summary, category_summary


# ============================================================
# 5. GENERATE AI BUSINESS INSIGHTS
# ============================================================

def generate_ai_report(
    kpis,
    region_summary,
    category_summary
):
    """Use Gemini to interpret Pandas-generated business statistics."""

    load_dotenv()

    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not gemini_api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found in the .env file."
        )


    # Gemini interprets the numbers.
    # Gemini does NOT calculate our core financial metrics.

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=gemini_api_key,
        temperature=0.2,
    )


    parser = JsonOutputParser(
        pydantic_object=BusinessReport
    )


    prompt = PromptTemplate(
        template="""
You are a business data analyst.

The statistics below were calculated using Python and Pandas
from a retail sales dataset.

You must use ONLY the supplied statistics.

Do not invent:
- financial figures
- regions
- product categories
- percentages
- trends that are not supported by the data

If some analysis is missing, simply do not make claims about it.

OVERALL BUSINESS KPIs:
{kpis}

REGION PERFORMANCE:
{region_summary}

CATEGORY PERFORMANCE:
{category_summary}

Based on this data:

1. Summarize overall business performance.
2. Identify the strongest area.
3. Identify the weakest area.
4. Identify important business insights.
5. Give practical management recommendations.

{format_instructions}
""",

        input_variables=[
            "kpis",
            "region_summary",
            "category_summary",
        ],

        partial_variables={
            "format_instructions":
                parser.get_format_instructions()
        },
    )


    # LCEL chain
    chain = prompt | model | parser


    result = chain.invoke(
        {
            "kpis": json.dumps(
                kpis,
                indent=2
            ),

            "region_summary": json.dumps(
                region_summary.to_dict(
                    orient="records"
                ),
                indent=2
            ),

            "category_summary": json.dumps(
                category_summary.to_dict(
                    orient="records"
                ),
                indent=2
            ),
        }
    )

    return result


# ============================================================
# 6. SAVE OUTPUTS
# ============================================================

def save_results(df, ai_report):
    """Save cleaned data and the AI-generated report."""

    df.to_csv(
        CLEANED_CSV_FILE,
        index=False
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            ai_report,
            file,
            indent=4,
            ensure_ascii=False
        )


    print("\nFILES SAVED")
    print("-" * 40)

    print(CLEANED_CSV_FILE)
    print(REPORT_FILE)


# ============================================================
# 7. MAIN PROGRAM
# ============================================================

def main():

    print("=" * 50)
    print("AI-POWERED BUSINESS DATA EXPLORER")
    print("=" * 50)


    try:

        # Step 1
        df = load_data(CSV_FILE)


        # Step 2
        df = clean_data(df)


        # Step 3
        kpis, region_summary, category_summary = (
            analyze_data(df)
        )


        print("\nOVERALL KPIs")
        print("-" * 40)

        for name, value in kpis.items():
            print(f"{name}: {value}")


        print("\nREGION ANALYSIS")
        print("-" * 40)

        print(
            region_summary.to_string(
                index=False
            )
        )


        # Don't call Gemini until you complete
        # your category_summary TODO.

        if category_summary.empty:

            print("\nPROJECT TODO")
            print("-" * 40)

            print(
                "Complete category_summary "
                "before generating the AI report."
            )

            return


        print("\nGENERATING AI REPORT...")
        print("-" * 40)


        # Step 4
        ai_report = generate_ai_report(
            kpis,
            region_summary,
            category_summary
        )


        print(
            json.dumps(
                ai_report,
                indent=2
            )
        )


        # Step 5
        save_results(
            df,
            ai_report
        )


        print("\nPROJECT COMPLETED SUCCESSFULLY.")


    except Exception as error:

        print("\nERROR:")
        print(error)


# Run main only when this file is executed directly
if __name__ == "__main__":
    main()