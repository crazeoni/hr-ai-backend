# # backend/hr_processor/rag_engine.py

# from hr_processor.embedder import query_index
# import os
# import openai
# from dotenv import load_dotenv

# load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# if not OPENAI_API_KEY:
#     raise RuntimeError("Missing OPENAI_API_KEY in .env")

# openai.api_key = OPENAI_API_KEY


# def build_prompt(question: str, contexts: list[str]):
#     joined = "\n\n---\n".join(contexts)

#     return f"""
# You are an HR Assistant AI.
# Use ONLY the provided HR policy text to answer the question.

# HR POLICY CONTEXT:
# {joined}

# QUESTION:
# {question}

# If the answer cannot be found in the policy, say:
# "I couldn't find this information in the HR document."
# """


# def generate_answer(question: str):
#     matches = query_index(question)
#     context_chunks = [m["metadata"]["text"] for m in matches]

#     prompt = build_prompt(question, context_chunks)

#     response = openai.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": prompt}]
#     )

#     return {
#         "answer": response["choices"][0]["message"]["content"],
#         "sources": context_chunks
#     }

# backend/hr_processor/rag_engine.py

from hr_processor.embedder import query_index
import os
import cohere
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not COHERE_API_KEY:
    raise RuntimeError("Missing COHERE_API_KEY in .env")

# Initialize Cohere client (reusing same client as embedder)
co = cohere.Client(COHERE_API_KEY)


def build_prompt(question: str, contexts: list[str]):
    joined = "\n\n---\n".join(contexts)

    return f"""You are an HR Assistant AI.
Use ONLY the provided HR policy text to answer the question.

HR POLICY CONTEXT:
{joined}

QUESTION:
{question}

If the answer cannot be found in the policy, say:
"I couldn't find this information in the HR document."
"""


def generate_answer(question: str):
    """
    Query the vector database and generate an answer using Cohere.
    Returns dict with 'answer' and 'sources'.
    """
    # Get relevant chunks from Pinecone
    matches = query_index(question)
    context_chunks = [m["metadata"]["text"] for m in matches]

    # Build the prompt
    prompt = build_prompt(question, context_chunks)

    # Use Cohere's chat endpoint to generate answer
    response = co.chat(
        model="command-r-08-2024",  # Fast and cost-effective; use "command-r-plus" for better quality
        message=prompt,
        temperature=0.3,  # Lower temperature for more factual responses
    )

    return {
        "answer": response.text,
        "sources": context_chunks
    }