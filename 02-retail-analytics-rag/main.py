import gradio as gr

from sql_analysis import (
    load_data,
    create_database,
    run_query
)

from visualization import sales_by_region_chart

from rag_llm import ask_policy


# -------------------------
# SETUP DATA
# -------------------------

df = load_data("data/retail_orders.csv")

create_database(df)


# -------------------------
# SQL
# -------------------------

def sql_section(query):

    try:
        return run_query(query)

    except Exception as error:
        return str(error)


# -------------------------
# VISUALIZATION
# -------------------------

def visualization_section():

    query = """
    SELECT
        region,
        SUM(quantity * unit_price) AS sales
    FROM orders
    GROUP BY region
    """

    result = run_query(query)

    return sales_by_region_chart(result)


# -------------------------
# GRADIO
# -------------------------

with gr.Blocks() as app:

    gr.Markdown(
        "# Retail Analytics + RAG Assistant"
    )

    with gr.Tab("SQL"):

        sql_input = gr.Textbox(
            label="SQL Query"
        )

        sql_button = gr.Button(
            "Run Query"
        )

        sql_output = gr.Dataframe()

        sql_button.click(
            sql_section,
            inputs=sql_input,
            outputs=sql_output
        )


    with gr.Tab("Visualization"):

        chart_button = gr.Button(
            "Generate Chart"
        )

        chart_output = gr.Plot()

        chart_button.click(
            visualization_section,
            outputs=chart_output
        )


    with gr.Tab("Policy RAG"):

        question = gr.Textbox(
            label="Ask a policy question"
        )

        rag_button = gr.Button(
            "Ask"
        )

        answer = gr.Textbox()

        rag_button.click(
            ask_policy,
            inputs=question,
            outputs=answer
        )


app.launch()