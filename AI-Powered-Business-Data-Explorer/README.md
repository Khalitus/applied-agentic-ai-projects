# AI-Powered Business Data Explorer

A small end-to-end analytics application that cleans retail transaction data with Pandas, calculates deterministic business metrics, and uses a Gemini model through LangChain to turn those verified statistics into a structured management report.

The project is intentionally built as a single `main.py` application. The focus is on making the full data-to-report workflow easy to follow before separating it into modules in a later iteration.

## Overview

The application processes a deliberately messy synthetic retail sales dataset and performs four main tasks:

1. Validates and cleans the source data.
2. Calculates business KPIs and grouped performance summaries with Pandas.
3. Sends only the calculated statistics—not the raw dataset—to the language model.
4. Saves both the cleaned dataset and a structured JSON business report.

The core design principle is simple:

> **Pandas calculates the numbers; the language model interprets them.**

This keeps financial calculations deterministic while still using an LLM where it is useful: summarization, interpretation, risk identification, and recommendations.

## Workflow

```text
Synthetic Retail CSV
        |
        v
Validation and Cleaning
        |
        v
Pandas Analytics
  - Overall KPIs
  - Region performance
  - Category performance
  - Top 5 products by sales
        |
        v
Structured Statistics
        |
        v
LangChain PromptTemplate
        |
        v
Gemini
        |
        v
JsonOutputParser
        |
        v
Structured Business Report
```

## Dataset

The included dataset is **synthetic/fabricated** and contains no real customer or company information.

The source file contains **612 rows and 10 columns**:

- `order_id`
- `order_date`
- `region`
- `category`
- `product`
- `customer`
- `quantity`
- `unit_price`
- `sales`
- `cost`

It was intentionally generated with common data-quality issues so the cleaning stage reflects a more realistic workflow. These include:

- missing values
- duplicate rows
- inconsistent capitalization and spacing
- mixed date formats
- numeric values stored as formatted text
- currency prefixes and comma-separated amounts

After cleaning, **591 usable rows** remain.

## Data Processing

The cleaning pipeline performs the following operations:

- validates the input file and required columns
- standardizes column names
- trims and normalizes text fields
- fills missing categorical values with `Unknown`
- converts financial columns to numeric values
- removes currency text and comma formatting
- converts dates with `pd.to_datetime()`
- removes duplicate rows
- removes rows with unusable financial values
- applies basic business-validity checks
- creates derived `profit` and `profit_margin` columns

## Business Analysis

All core business metrics are calculated directly in Python and Pandas.

### Overall KPIs

The current dataset produces:

| Metric | Result |
|---|---:|
| Total Orders | 591 |
| Total Sales | 67,580,017.80 |
| Total Cost | 50,178,983.78 |
| Total Profit | 17,401,034.02 |
| Average Order Value | 114,348.59 |
| Overall Profit Margin | 25.75% |
| Total Quantity Sold | 2,075 |

### Regional Analysis

The application groups transactions by region and calculates:

- total sales
- total profit
- total quantity
- total orders
- profit margin

Karachi is the largest region in the current synthetic dataset by both sales and profit.

### Category Analysis

The same analysis is performed by product category.

Electronics generates the highest sales and total profit, while Clothing records the highest category profit margin in the current dataset.

### Top Products

The application also identifies the five products with the highest total sales:

| Product | Total Sales |
|---|---:|
| Laptop | 15,240,935.52 |
| Smartphone | 9,909,646.96 |
| Monitor | 6,892,061.53 |
| Study Desk | 5,281,674.25 |
| Filing Cabinet | 4,332,262.36 |

## AI Reporting Layer

The application uses LangChain to pass the Pandas-generated statistics to Gemini.

The model does not receive the full raw dataset. Instead, the application converts the calculated KPI, regional, and category summaries into structured JSON-compatible data and supplies them to a `PromptTemplate`.

The LCEL pipeline is:

```python
chain = prompt | model | parser
```

`JsonOutputParser` is used with a Pydantic schema so that the final response follows a predictable structure:

```text
executive_summary
strongest_area
weakest_area
key_insights
risks
recommendations
```

The prompt also instructs the model to use only supplied statistics and not invent financial figures, categories, regions, percentages, or unsupported trends.

## Example Report

A typical generated report identifies:

- Karachi as the strongest region by sales and profit
- Electronics as the largest category by sales
- Clothing as the highest-margin category
- Multan as the highest-margin region
- `Unknown` region/category values as a data-quality concern
- concentration in Karachi and Electronics as a potential business risk

The full generated response is written to `business_report.json`.

## Project Structure

```text
AI-Powered-Business-Data-Explorer/
|
|-- main.py
|-- fake_retail_sales_messy.csv
|-- requirements.txt
|-- .gitignore
|
|-- cleaned_retail_sales.csv      # generated after execution
|-- business_report.json          # generated after execution
|
`-- .env.local                    # local only; contains API key
```

## Tech Stack

- Python
- Pandas
- NumPy
- Pydantic
- LangChain
- Google Gemini
- python-dotenv

## Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd AI-Powered-Business-Data-Explorer
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a `.env.local` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

The environment file is excluded from version control and should never be committed.

## Running the Project

Run:

```bash
python main.py
```

The program will:

1. load and validate the CSV
2. print the original data-quality summary
3. clean and transform the dataset
4. calculate KPIs
5. print regional and category analysis
6. print the top five products by sales
7. generate the structured AI report
8. save the output files

## Output Files

### `cleaned_retail_sales.csv`

Contains the cleaned transaction data together with the derived:

- `profit`
- `profit_margin`

columns.

### `business_report.json`

Contains the structured LLM-generated business interpretation, including the executive summary, strongest and weakest areas, key insights, risks, and recommendations.

## Design Decisions

### Deterministic analytics before LLM interpretation

The model is not responsible for calculating core financial KPIs. Pandas performs the calculations first, and Gemini receives the resulting statistics.

This reduces the risk of using probabilistic model output for calculations that can be handled reliably in code.

### Single-file implementation

The first version intentionally keeps the workflow in `main.py`. This makes the execution path easy to inspect while learning the stack.

A larger production application would normally separate data loading, cleaning, analysis, AI reporting, and configuration into individual modules.

### Synthetic source data

The repository uses fabricated retail data so the project can be run and reviewed without exposing real customer, financial, or commercial information.

## Limitations

This is a learning and portfolio project rather than a production analytics platform.

Current limitations include:

- input schema is fixed to the expected retail CSV structure
- data-quality rules are intentionally simple
- the application is command-line based
- LLM output can still require human review
- no database, API, dashboard, or deployment layer is included
- generated recommendations are based only on the supplied aggregate statistics

## Purpose

This project was built to consolidate practical skills across Python, Pandas, structured data analysis, prompt engineering, LangChain LCEL, Pydantic schemas, and structured LLM output in one complete workflow.
