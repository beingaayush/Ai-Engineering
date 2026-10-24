# Yahi main evaluation file hai. Ismein 5 metrics calculate honge:
# - Context Precision
# - Context Recall
# - Faithfulness
# - Answer Relevancy
# - Answer Correctness
# -----------------------------------------------------------------

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from main import answer_question
from retrieval import setup_knowledgebase

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset.json"
RESULTS_PATH = BASE_DIR / "results.json"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Set GROQ_API_KEY in the project's .env file.")

groq_client = Groq(api_key=GROQ_API_KEY)


def llm_judge(prompt):
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    return json.loads(response.choices[0].message.content)


def context_precision(question, retrieved_docs):
    if not retrieved_docs:
        return 0.0

    relevant_chunks = 0

    for doc in retrieved_docs:
        chunk = (doc.payload or {}).get("text", "")

        prompt = f"""
You are evaluating retrieval quality in a RAG system.

Question: {question}
Retrieved chunk: {chunk}

Is this chunk relevant to answering the question?

Return JSON only:
{{"relevant": true, "reason": "short explanation"}}

Set relevant to true only if the chunk contains useful
information for answering the question.
"""

        result = llm_judge(prompt)

        if result.get("relevant") is True:
            relevant_chunks += 1

    # Simplified precision, not the standard rank-aware formula.
    return relevant_chunks / len(retrieved_docs)


def context_recall(question, context, ground_truth):
    prompt = f"""
You are evaluating retrieval quality in a RAG system.

Question: {question}
Ground truth answer: {ground_truth}
Retrieved context: {context}

Does the context contain enough information to produce
the ground-truth answer?

Return JSON only:
{{"score": 0.0, "reason": "short explanation"}}

Scoring:
1.0 = All important information is present.
0.7 = Most important information is present.
0.5 = Some important information is present.
0.0 = Required information is absent.
"""

    return llm_judge(prompt)


def evaluate_faithfulness(question, context, answer):
    prompt = f"""
You are evaluating a RAG system.

Question: {question}
Retrieved context: {context}
Generated answer: {answer}

Are the claims in the answer supported by the context?

Return JSON only:
{{"score": 0.0, "reason": "short explanation"}}

Scoring:
1.0 = All claims are supported.
0.7 = Mostly supported with minor issues.
0.5 = Some claims are supported.
0.0 = Unsupported or contradictory.
"""

    return llm_judge(prompt)


def evaluate_relevancy(question, answer):
    prompt = f"""
You are evaluating a RAG system.

Question: {question}
Generated answer: {answer}

Does the answer actually answer the question?

Return JSON only:
{{"score": 0.0, "reason": "short explanation"}}

Scoring:
1.0 = Directly answers the question.
0.7 = Mostly answers the question.
0.5 = Partially answers the question.
0.0 = Completely off-topic.
"""

    return llm_judge(prompt)


def evaluate_correctness(answer, ground_truth):
    prompt = f"""
You are evaluating a RAG system.

Generated answer: {answer}
Ground-truth answer: {ground_truth}

Is the generated answer factually correct compared
with the ground truth?

Return JSON only:
{{"score": 0.0, "reason": "short explanation"}}

Scoring:
1.0 = Completely correct.
0.7 = Mostly correct with minor omissions.
0.5 = Partially correct.
0.0 = Incorrect or contradictory.
"""

    return llm_judge(prompt)


def score_value(result):
    score = float(result["score"])
    return max(0.0, min(1.0, score))


def run_evaluation():
    setup_knowledgebase()

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    metric_names = [
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
    ]

    score_lists = {
        metric: [] for metric in metric_names
    }

    case_results = []

    for test in dataset:
        question = test["question"]
        ground_truth = test["ground_truth"]

        # Run the actual RAG pipeline.
        rag_result = answer_question(question, top_k=3)

        retrieved_docs = rag_result["retrieved_docs"]
        context = rag_result["context"]
        answer = rag_result["answer"]

        # Evaluate the retrieval and generated answer.
        precision = context_precision(
            question, retrieved_docs
        )

        recall_result = context_recall(
            question, context, ground_truth
        )

        faithfulness_result = evaluate_faithfulness(
            question, context, answer
        )

        relevancy_result = evaluate_relevancy(
            question, answer
        )

        correctness_result = evaluate_correctness(
            answer, ground_truth
        )

        scores = {
            "context_precision": precision,
            "context_recall": score_value(recall_result),
            "faithfulness": score_value(faithfulness_result),
            "answer_relevancy": score_value(relevancy_result),
            "answer_correctness": score_value(correctness_result),
        }

        for metric, score in scores.items():
            score_lists[metric].append(score)

        case_results.append({
            "id": test["id"],
            "question": question,
            "ground_truth": ground_truth,
            "expected_information": test.get(
                "expected_information"
            ),
            "retrieved_context": context,
            "generated_answer": answer,
            "scores": scores,
            "judge_reasons": {
                "context_recall": recall_result.get("reason", ""),
                "faithfulness": faithfulness_result.get("reason", ""),
                "answer_relevancy": relevancy_result.get("reason", ""),
                "answer_correctness": correctness_result.get("reason", ""),
            },
        })

        print(f"\nQUESTION: {question}")
        print(f"ANSWER: {answer}")

        for metric, score in scores.items():
            print(f"{metric}: {score:.2f}")

        low_metrics = [
            metric
            for metric, score in scores.items()
            if score < 0.7
        ]

        if low_metrics:
            print("Needs review:", ", ".join(low_metrics))
        else:
            print("All metrics are at least 0.70.")

    # Calculate average score for every metric.
    averages = {
        metric: (
            sum(values) / len(values)
            if values else 0.0
        )
        for metric, values in score_lists.items()
    }

    output = {
        "dataset_size": len(dataset),
        "metric_averages": averages,
        "test_cases": case_results,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nAVERAGE SCORES")

    for metric, score in averages.items():
        print(f"{metric}: {score:.2f}")

    print(f"\nDetailed results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    run_evaluation()