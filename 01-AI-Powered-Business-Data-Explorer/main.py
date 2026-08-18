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