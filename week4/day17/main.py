# Ye actual RAG application hai: question → retrieval → LLM answer.
# -----------------------------------------------------------------


import os

from dotenv import load_dotenv
from groq import Groq

from retrieval import search, setup_knowledgebase

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Set GROQ_API_KEY in your .env file.")

groq_client = Groq(api_key=GROQ_API_KEY)


def ask_llm(question, context):
    prompt = f"""
Answer the question using only the information in the context.

Context:
{context}

Question:
{question}

If the answer is not available in the context, say:
"I don't know based on the provided information."
"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    return response.choices[0].message.content


def answer_question(question, top_k=3):
    retrieved_docs = search(question, top_k=top_k)

    context = "\n".join(
        result.payload.get("text", "")
        for result in retrieved_docs
        if result.payload
    )

    answer = ask_llm(question, context)

    return {
        "question": question,
        "retrieved_docs": retrieved_docs,
        "context": context,
        "answer": answer,
    }


def main():
    count = setup_knowledgebase()

    print(f"Knowledgebase ready: {count} documents indexed.")
    print("Ask a question. Type 'exit' to quit.")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        result = answer_question(question)

        print("\nRetrieved Context:")
        print(result["context"])

        print("\nAnswer:")
        print(result["answer"])


if __name__ == "__main__":
    main()