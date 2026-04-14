"""
RAG Evaluation Script — Uses Ragas to evaluate the LangGraph RAG pipeline.

Usage:
  1. Make sure you have a session active (upload a PDF via the API first)
  2. Run: python evaluation/run_eval.py --session_id <YOUR_SESSION_ID>
  
  Or run standalone (it will upload dummy.pdf automatically):
     python evaluation/run_eval.py
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Add parent directory to path so we can import rag_engine
sys.path.insert(0, str(Path(__file__).parent.parent))

import rag_engine


def load_eval_dataset(filepath: str = None) -> list:
    """Load evaluation questions from JSON file."""
    if filepath is None:
        filepath = Path(__file__).parent / "eval_dataset.json"
    
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(session_id: str, eval_data: list) -> dict:
    """Run each question through the RAG pipeline and collect results."""
    results = []
    
    for i, item in enumerate(eval_data):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        
        print(f"\n[{i+1}/{len(eval_data)}] Evaluating: {question[:60]}...")
        
        try:
            response = rag_engine.ask_question(session_id, question)
            
            result = {
                "question": question,
                "answer": response["answer"],
                "ground_truth": ground_truth,
                "contexts": [c["text"] for c in response.get("source_chunks", [])],
                "expanded_query": response.get("expanded_query", ""),
                "graph_metadata": response.get("graph_metadata", {}),
            }
            results.append(result)
            print(f"   ✅ Answer: {response['answer'][:80]}...")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                "question": question,
                "answer": f"ERROR: {e}",
                "ground_truth": ground_truth,
                "contexts": [],
                "error": str(e),
            })
    
    return results


def compute_ragas_scores(results: list) -> dict:
    """Compute Ragas evaluation metrics."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        
        # Prepare data in Ragas format
        eval_data = {
            "question": [r["question"] for r in results if "error" not in r],
            "answer": [r["answer"] for r in results if "error" not in r],
            "contexts": [r["contexts"] for r in results if "error" not in r],
            "ground_truth": [r["ground_truth"] for r in results if "error" not in r],
        }
        
        if not eval_data["question"]:
            return {"error": "No valid results to evaluate"}
        
        dataset = Dataset.from_dict(eval_data)
        
        print("\n🔍 Running Ragas evaluation (this may take a minute)...")
        eval_result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )
        
        scores = {
            "faithfulness": round(eval_result["faithfulness"], 4),
            "answer_relevancy": round(eval_result["answer_relevancy"], 4),
            "context_precision": round(eval_result["context_precision"], 4),
        }
        
        return scores
        
    except ImportError as e:
        print(f"⚠️  Ragas evaluation requires additional setup: {e}")
        print("   Install with: pip install ragas datasets")
        return {"error": str(e)}
    except Exception as e:
        print(f"⚠️  Ragas evaluation failed: {e}")
        return {"error": str(e)}


def save_results(results: list, scores: dict, output_path: str = None):
    """Save evaluation results to JSON."""
    if output_path is None:
        output_path = Path(__file__).parent / "results.json"
    
    output = {
        "scores": scores,
        "num_questions": len(results),
        "num_errors": sum(1 for r in results if "error" in r),
        "results": results,
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline using Ragas metrics")
    parser.add_argument("--session_id", type=str, help="Session ID to evaluate against")
    parser.add_argument("--pdf", type=str, default="dummy.pdf", help="PDF file to upload if no session_id")
    parser.add_argument("--dataset", type=str, help="Path to custom eval dataset JSON")
    parser.add_argument("--skip-ragas", action="store_true", help="Skip Ragas metrics, just run pipeline")
    
    args = parser.parse_args()
    
    # Load evaluation dataset
    eval_data = load_eval_dataset(args.dataset)
    print(f"📋 Loaded {len(eval_data)} evaluation questions")
    
    # Get or create session
    session_id = args.session_id
    if not session_id:
        pdf_path = Path(__file__).parent.parent / args.pdf
        if not pdf_path.exists():
            print(f"❌ PDF not found: {pdf_path}")
            print("   Please provide --session_id or --pdf path")
            sys.exit(1)
        
        print(f"📄 Uploading {pdf_path.name}...")
        with open(pdf_path, "rb") as f:
            result = rag_engine.create_session(f.read(), pdf_path.name)
        session_id = result["session_id"]
        print(f"   Session created: {session_id} ({result['total_chunks']} chunks)")
    
    # Run pipeline on all questions
    print(f"\n🚀 Running evaluation on session {session_id}...")
    results = run_evaluation(session_id, eval_data)
    
    # Compute Ragas scores
    scores = {}
    if not args.skip_ragas:
        scores = compute_ragas_scores(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Questions evaluated: {len(results)}")
    print(f"  Errors: {sum(1 for r in results if 'error' in r)}")
    
    if scores and "error" not in scores:
        print(f"\n  📈 Ragas Scores:")
        print(f"     Faithfulness:      {scores.get('faithfulness', 'N/A')}")
        print(f"     Answer Relevancy:  {scores.get('answer_relevancy', 'N/A')}")
        print(f"     Context Precision: {scores.get('context_precision', 'N/A')}")
    elif scores.get("error"):
        print(f"\n  ⚠️  Ragas scoring failed: {scores['error']}")
    
    print("=" * 60)
    
    # Save
    save_results(results, scores)
    
    return results, scores


if __name__ == "__main__":
    main()
