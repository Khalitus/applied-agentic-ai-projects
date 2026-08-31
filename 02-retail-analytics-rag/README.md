# Retail Analytics + RAG Assistant

A small end-to-end retail analytics application that combines structured business analysis with retrieval-augmented generation (RAG).

The project uses retail order and product data for SQL-based analytics and visualization, while a separate company-policy knowledge base is indexed in Chroma and queried through Gemini. A Gradio interface brings both paths together in one application.

The project is intentionally scoped as a learning and portfolio project. The data and company policies are synthetic.

## Problem

Business information often exists in two different forms:

- **Structured data** such as orders, products, prices, discounts, regions, and returns.
- **Unstructured knowledge** such as return rules, delivery timelines, approval limits, and customer-service policies.

Structured questions such as:

> Which region generated the most sales?

are better handled with SQL.

Policy questions such as:

> Who approves a 12% discount?

require searching unstructured text and grounding the answer in the relevant policy.

Using only an LLM for both creates unnecessary risk: numerical analysis becomes less reliable, and policy answers can be produced without evidence from the actual company document.

## Solution

This project separates those responsibilities.

The structured-data path loads and cleans two CSV files, validates business rules, stores the data in SQLite, and runs predefined SQL analyses for sales, profitability, delivery, returns, and customer behavior.

The unstructured-data path loads a company-policy document, splits it into chunks, converts those chunks into embeddings, stores them in Chroma, retrieves the most relevant chunks for a question, and sends only that retrieved context to Gemini.

Gradio provides a simple interface with three areas:

- **Analytics** — run predefined business analyses and inspect the returned DataFrames.
- **Visualizations** — generate charts from SQL results or cleaned Pandas data.
- **Policy assistant** — ask questions against the policy knowledge base and receive grounded answers with source chunk information.

## Architecture

```text
Structured data

retail_orders.csv + products.csv
              |
              v
           Pandas
              |
        clean + validate
              |
              v
           SQLite
              |
              v
        SQL analytics
          /       \
         v         v
    DataFrames   Matplotlib/
                 Seaborn
          \       /
           v     v
            Gradio


Unstructured knowledge

company_policies.txt
        |
        v
    TextLoader
        |
        v
CharacterTextSplitter
        |
        v
Hugging Face embeddings
        |
        v
      Chroma
        |
        v
 semantic retriever
        |
        v
retrieved policy context
        |
        v
      Gemini
        |
        v
answer + source metadata
        |
        v
      Gradio
```

## Features

### Business analytics

The SQL layer includes analyses for:

- Overall order, unit, sales, and average-order-value KPIs
- Sales by region
- Region performance
- Channel performance
- Monthly sales
- Product profitability
- Category profitability
- Supplier contribution
- Return rate by category
- City order summaries
- Recent orders
- South-region online orders
- Largest orders
- High-discount orders
- Low-rated orders

The relational analyses join the order and product tables through `product_id` to calculate cost, profit, and profit margin.

### Data validation

Before the SQLite database is created, the application checks:

- Non-positive quantities
- Non-positive prices
- Discounts outside the valid `0–1` range
- Ratings outside the valid `1–5` range
- Order product IDs missing from the product master
- Duplicate order IDs
- Duplicate product IDs

Missing customer segments are labeled as `Unknown`.

Missing ratings are intentionally left missing rather than being replaced with an invented average rating.

### Visualizations

The application includes:

- Net sales by region — bar chart
- Monthly net sales — line chart
- Profit by product category — bar chart
- Discount vs order value — scatter plot
- Customer rating distribution — histogram
- Region × category net-sales heatmap
- Return rate by category — bar chart

The visualization layer consumes either prepared SQL results or cleaned Pandas DataFrames instead of duplicating business calculations inside plotting code.

### Policy RAG assistant

The RAG pipeline:

1. Loads `company_policies.txt`
2. Splits it into overlapping text chunks
3. Attaches source and chunk metadata
4. Embeds the chunks using `sentence-transformers/all-MiniLM-L6-v2`
5. Stores the embeddings in Chroma
6. Retrieves the three most relevant chunks for a question
7. Combines retrieved text into the prompt context
8. Sends the grounded prompt to Gemini
9. Returns the answer with retrieved source metadata

The prompt instructs the model to use only the supplied policy context and to preserve policy-specific numbers, approval levels, percentages, and time periods.

If the available context does not support an answer, the expected response is:

```text
I don't know based on the available company policies.
```

Unsupported answers are returned without source attribution rather than attaching unrelated retrieved chunks.

### Gradio interface

The interface includes:

- Four KPI cards
- Analytics dropdown with tabular output
- Visualization dropdown with rendered Matplotlib figures
- Policy-question textbox
- Example policy questions
- Source-aware RAG answers

On startup, the application rebuilds the local Chroma index automatically if it does not already exist.

## Tech Stack

| Technology | Responsibility |
|---|---|
| Python | Application logic and orchestration |
| Pandas | CSV loading, cleaning, validation, and in-memory analysis |
| SQLite | Local relational database for structured retail data |
| SQL | Filtering, aggregation, joins, profitability, and business analysis |
| Matplotlib | Figure creation and chart control |
| Seaborn | Statistical and business visualizations |
| LangChain | Document loading, splitting, retrieval, and model integration |
| Hugging Face Sentence Transformers | Local text embeddings |
| Chroma | Persistent local vector store |
| Gemini | Grounded natural-language policy responses |
| Gradio | Interactive application interface |
| python-dotenv | Local environment-variable loading |

## Dataset

The project uses synthetic retail data.

### `retail_orders.csv`

- **900 orders**
- **14 columns**
- Date range: **2025-01-02 to 2025-12-31**
- 4 regions
- 3 sales channels
- Contains order quantity, price, discount, delivery, rating, return status, customer segment, payment method, and product references

The raw dataset includes a small amount of missing data for cleaning practice:

- 14 missing customer-segment values
- 11 missing ratings

### `products.csv`

- **12 products**
- **7 columns**
- Contains product name, category, list price, unit cost, supplier, and warranty information

The `product_id` field connects the product master to the order table.

### `company_policies.txt`

The synthetic policy knowledge base covers:

- Returns and refunds
- Shipping and delivery
- Discount authorization
- Customer service

## How It Works

### Application startup

When `main.py` starts:

```text
Load CSV files
      |
      v
Clean data
      |
      v
Validate business rules
      |
      v
Create SQLite tables
      |
      v
Check for Chroma index
      |
      +---- exists ----> continue
      |
      +---- missing ---> load policy -> split -> embed -> index
      |
      v
Launch Gradio
```

This keeps generated database files separate from the source data and allows the project to rebuild them locally.

### Structured analytics flow

For a request such as **Product profitability**:

```text
Gradio dropdown
      |
      v
product_profitability()
      |
      v
SQLite JOIN + aggregation
      |
      v
Pandas DataFrame
      |
      v
Gradio table
```

The query joins `orders` and `products` to calculate:

```text
Net sales = quantity × unit price × (1 - discount)

Cost = quantity × unit cost

Profit = net sales - cost

Profit margin = profit / net sales × 100
```

### Visualization flow

For **Monthly sales**:

```text
monthly_sales()
      |
      v
12-row SQL result
      |
      v
plot_monthly_sales()
      |
      v
Seaborn line chart
      |
      v
Gradio Plot
```

SQL handles the aggregation; the plotting layer focuses on presentation.

### RAG flow

For:

```text
Who approves a 12 percent discount?
```

the application performs:

```text
Question
   |
   v
Embedding
   |
   v
Chroma similarity search
   |
   v
Relevant policy chunks
   |
   v
Grounded prompt
   |
   v
Gemini
   |
   v
Answer + retrieved sources
```

The policy states that discounts above 10% and up to 15% require Commercial Head approval, so the assistant can answer from retrieved company-policy context instead of relying on general model knowledge.

## Project Structure

```text
retail-analytics-rag/
├── main.py
├── sql_analysis.py
├── visualization.py
├── rag_llm.py
│
├── data/
│   ├── retail_orders.csv
│   ├── products.csv
│   └── company_policies.txt
│
├── output/
│
├── requirements.txt
├── .gitignore
└── .env.local
```

Generated locally:

```text
data/retail.db
data/chroma_db/
```

These files can be rebuilt from the source data and are intended to remain outside version control.

`.env.local` contains the API key and should also remain outside version control.

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd retail-analytics-rag
```

### 2. Create and activate a virtual environment

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Gemini

Create a `.env.local` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-3.5-flash
```

Make sure `.env.local` is included in `.gitignore`.

A minimal `.gitignore` should include:

```gitignore
.venv/
__pycache__/
*.pyc
.env
.env.local
data/retail.db
data/chroma_db/
.DS_Store
```

## Usage

Run:

```bash
python main.py
```

Gradio will print a local URL, normally:

```text
http://127.0.0.1:7860
```

The current application also launches with `share=True`, so Gradio can provide a temporary public share link while the local Python process remains running.

The share link is not a permanent deployment.

### Analytics

Open the **Analytics** tab, select an analysis, and click **Run analysis**.

Examples:

```text
Region performance
Product profitability
Monthly sales
Return rate by category
```

### Visualizations

Open the **Visualizations** tab and choose a chart.

Examples:

```text
Monthly sales
Category profitability
Region-category heatmap
Discount vs order value
```

### Policy assistant

Example supported questions:

```text
Who approves a 12 percent discount?

How long does delivery take in the West region?

Can an opened non-defective product be returned?

What happens when a customer gives a rating of 2?

When does a delayed shipment need to be escalated?
```

An intentionally unsupported question such as:

```text
How many vacation days do employees receive?
```

should not be answered from outside knowledge.

## Example Output

The finalized synthetic dataset produces the following overall KPIs:

| Metric | Result |
|---|---:|
| Total orders | 900 |
| Units sold | 2,551 |
| Net sales | 81,491,821.66 |
| Average order value | 90,546.47 |

Selected analysis results:

| Finding | Result |
|---|---:|
| Highest-sales region | South — 29,706,709.26 |
| Highest-sales channel | Online — 33,723,292.41 |
| Highest average order value by channel | Retail Store — 105,921.47 |
| Highest-sales month | August 2025 — 8,636,873.95 |
| Highest-profit product | Nimbus Pro Laptop — 2,381,888.76 |
| Highest-profit category | Computers — 6,510,225.54 |
| Highest-margin category | Accessories — 36.82% |
| Highest-return-rate category | Computers — 5.93% |
| Total profit across all categories | 17,926,721.65 |

A useful result from the profitability analysis is that the highest-sales or highest-volume group is not always the highest-margin group. The project therefore exposes multiple KPIs rather than treating one metric as a complete measure of performance.

Example policy question:

```text
Question:
Who approves a 12 percent discount?

Answer:
A 12% discount requires Commercial Head approval.
```

Supported answers also include source information from the retrieved policy chunks.

## Key Learnings

### Structured and unstructured data need different tools

SQL is used where calculations need to be deterministic and reproducible.

RAG is used where the information exists in text and must first be retrieved before an LLM can answer.

Keeping those responsibilities separate makes the project easier to reason about and test.

### Cleaning and validation are different steps

Cleaning repairs known data-quality issues.

Validation checks whether the resulting data obeys business rules.

The project validates quantities, prices, discounts, ratings, identifiers, and table relationships before creating the database.

### Missing values should not always be imputed

Missing customer segments can reasonably be labeled `Unknown`.

Missing ratings are different: replacing an absent rating with an average rating would invent customer feedback and distort later rating analysis.

They are therefore kept as missing values.

### SQL performs the business calculations

Sales, cost, profit, profit margin, return rate, monthly summaries, joins, and grouped performance calculations are handled in SQL.

Visualization functions receive already-prepared data rather than recreating the same business logic.

### RAG quality depends on retrieval quality

The language model can only ground its answer in the context that retrieval provides.

The project therefore builds RAG in explicit stages:

```text
Load
→ Split
→ Embed
→ Store
→ Retrieve
→ Augment
→ Generate
```

Inspecting chunks and retrieval behavior during development was useful for separating retrieval problems from generation problems.

### Generated data should be reproducible

`retail.db` and the Chroma index are generated artifacts rather than source files.

The application creates the SQLite database at startup and rebuilds the vector index when it is missing.

## Limitations

This project is deliberately small in scope.

- The retail data and policies are synthetic and do not represent a real company.
- Analytics are predefined; the application does not translate arbitrary natural-language questions into SQL.
- The RAG system uses one text policy document rather than a multi-document ingestion pipeline.
- Retrieval uses similarity search with a fixed `k=3`; there is no reranker, hybrid search, or calibrated relevance threshold.
- Source reporting identifies retrieved chunks but does not provide sentence-level citation spans.
- Chroma and SQLite are local stores rather than shared production databases.
- The Gemini API is required for policy-answer generation.
- There is no authentication or user-level authorization.
- The Gradio share link is temporary and is not a production deployment.
- The project currently relies mainly on manual functional testing rather than a dedicated automated test suite.
- Embedding, vector-store, and LLM objects could be cached more aggressively for better repeated-request performance.

These limitations are acceptable for the current learning scope but would need to be addressed before using a similar design in a real business system.

## Future Improvements

Reasonable next steps would be:

- Add automated tests for data validation, SQL results, and RAG behavior
- Add multi-document policy ingestion
- Support PDF and other document formats
- Add retrieval evaluation and relevance thresholds
- Evaluate reranking or hybrid retrieval when the knowledge base becomes larger
- Show retrieved source snippets more clearly in the interface
- Cache the embedding model, retriever, vector store, and LLM where appropriate
- Add user-selectable filters for region, category, channel, and date range
- Add safe natural-language-to-SQL only after the structured analytics layer is well tested
- Deploy the application to a persistent hosting environment instead of relying on a temporary Gradio share link

The current version intentionally stops before agents, autonomous routing, or more complex orchestration because those are not necessary to solve the project’s present problem.
