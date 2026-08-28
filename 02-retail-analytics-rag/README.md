# Retail Analytics & Policy Assistant

Month-1 capstone combining:

- Python / Pandas
- Intro SQL
- Data Visualization
- RAG
- Gradio

## The business idea

Different questions should use different data tools.

```text
“How much did South sell?”        -> SQL
“How did monthly sales change?”   -> Visualization
“Who approves a 12% discount?”    -> RAG
```

That distinction is the point of the project.

## Architecture

```text
                     RETAIL INTELLIGENCE APP
                              |
              +---------------+----------------+
              |               |                |
              v               v                v
          SQL Analytics   Visualization     Policy RAG
              |               |                |
              v               v                v
        SQLite tables      SQL/Pandas       TXT documents
              |               |                |
              |               |             Load/Split
              |               |                |
              |               |              Embed
              |               |                |
              |               |              Chroma
              |               |                |
              |               |             Retrieve
              |               |                |
              +---------------+-------+--------+
                                      |
                                      v
                                  User result
```

## Data

- `retail_orders.csv`: 900 fictional 2025 order lines
- `products.csv`: 12 fictional products
- four fictional policy documents for RAG

There is no real customer information.

## Setup

### 1. Virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install

```bash
pip install -r requirements.txt
```

### 3. Build database

```bash
python build_db.py
```

### 4. Verify

```bash
python check_project.py
```

Expected:

```text
orders                : 900
products              : 12
duplicate_order_ids   : 0
regions               : 4
```

### 5. RAG key

SQL and visualization work without an LLM.

For RAG:

```bash
cp .env.example .env
```

Add your key and model name to `.env`.

### 6. Run

```bash
python app.py
```

## Scope boundary

Do **not** add natural-language-to-SQL or autonomous routing yet.

Those require tool calling / agents, which belong later in the roadmap.

For Month 1, understand each subsystem separately.

## Definition of done

You are finished when you can:

- rebuild the database from CSV
- explain both table schemas
- write 8+ useful SQL queries
- use INNER JOIN correctly
- build 5 purposeful charts
- explain why each chart was chosen
- trace RAG from source file to final answer
- show source evidence
- demonstrate an unanswerable RAG question without hallucination
- run the Gradio app locally
- draw the architecture from memory

Suggested repo name:

`retail-analytics-rag`
