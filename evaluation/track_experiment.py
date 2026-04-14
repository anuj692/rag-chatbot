"""
MLflow Experiment Tracker — Logs RAG pipeline parameters and Ragas evaluation metrics.

Usage:
  python evaluation/track_experiment.py --pdf dummy.pdf
  python evaluation/track_experiment.py --session_id <ID> --experiment-name "chunk-size-test"

After running, view results with:
  mlflow ui
  (opens at http://localhost:5000)
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_experiment(
    pdf_path: str = None,
    session_id: str = None,
    experiment_name: str = "RAG_Pipeline_Evaluation",
    eval_dataset_path: str = None,
):
    """Run a full evaluation + MLflow tracking experiment."""
    
    import mlflow
    from run_eval import load_eval_dataset, run_evaluation, compute_ragas_scores, save_results
    import rag_engine
    from rag_engine import RAG_CONFIG
    from rag_graph import LLM_MODEL, LLM_TEMPERATURE, QUERY_EXPANSION_PROMPT, ANSWER_GENERATION_PROMPT, DOCUMENT_GRADER_PROMPT
    
    # Set up MLflow
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # ─── Log Parameters ──────────────────────────────────────────────
        mlflow.log_params({
            "embedding_model": RAG_CONFIG["embedding_model"],
            "llm_model": RAG_CONFIG["llm_model"],
            "chunk_size": RAG_CONFIG["chunk_size"],
            "chunk_overlap": RAG_CONFIG["chunk_overlap"],
            "top_k_final": RAG_CONFIG["top_k_final"],
            "ensemble_weight_bm25": RAG_CONFIG["ensemble_weights"][0],
            "ensemble_weight_faiss": RAG_CONFIG["ensemble_weights"][1],
            "temperature": LLM_TEMPERATURE,
            "pipeline_type": "LangGraph_StateGraph",
        })
        
        # Log prompt templates as artifacts
        prompts_dir = Path(__file__).parent / "mlflow_prompts"
        prompts_dir.mkdir(exist_ok=True)
        
        (prompts_dir / "query_expansion.txt").write_text(QUERY_EXPANSION_PROMPT.template)
        (prompts_dir / "answer_generation.txt").write_text(ANSWER_GENERATION_PROMPT.template)
        (prompts_dir / "document_grader.txt").write_text(DOCUMENT_GRADER_PROMPT.template)
        mlflow.log_artifacts(str(prompts_dir), "prompt_templates")
        
        # ─── Create Session ──────────────────────────────────────────────
        if not session_id:
            pdf_file = Path(__file__).parent.parent / (pdf_path or "dummy.pdf")
            if not pdf_file.exists():
                print(f"❌ PDF not found: {pdf_file}")
                sys.exit(1)
            
            print(f"📄 Uploading {pdf_file.name}...")
            with open(pdf_file, "rb") as f:
                result = rag_engine.create_session(f.read(), pdf_file.name)
            session_id = result["session_id"]
            mlflow.log_param("pdf_filename", pdf_file.name)
            mlflow.log_param("total_chunks", result["total_chunks"])
            mlflow.log_param("total_pages", result["total_pages"])
            print(f"   Session: {session_id} ({result['total_chunks']} chunks)")
        
        # ─── Run Evaluation ──────────────────────────────────────────────
        eval_data = load_eval_dataset(eval_dataset_path)
        mlflow.log_param("num_eval_questions", len(eval_data))
        
        print(f"\n🚀 Running pipeline evaluation...")
        results = run_evaluation(session_id, eval_data)
        
        num_errors = sum(1 for r in results if "error" in r)
        mlflow.log_metric("num_errors", num_errors)
        
        # ─── Compute Ragas Scores ────────────────────────────────────────
        print("\n📊 Computing Ragas metrics...")
        scores = compute_ragas_scores(results)
        
        if scores and "error" not in scores:
            for metric_name, score_value in scores.items():
                mlflow.log_metric(metric_name, score_value)
                print(f"   {metric_name}: {score_value}")
        else:
            print(f"   ⚠️  Ragas scoring issue: {scores.get('error', 'unknown')}")
            mlflow.log_param("ragas_error", scores.get("error", "unknown"))
        
        # ─── Log Graph Metadata ──────────────────────────────────────────
        # Average timing per node across all evaluations
        timing_sums = {}
        timing_counts = {}
        for r in results:
            meta = r.get("graph_metadata", {})
            for key in ["expand_query_time", "retrieve_time", "grade_time", "generate_time"]:
                if key in meta:
                    timing_sums[key] = timing_sums.get(key, 0) + meta[key]
                    timing_counts[key] = timing_counts.get(key, 0) + 1
        
        for key in timing_sums:
            avg_time = round(timing_sums[key] / timing_counts[key], 3)
            mlflow.log_metric(f"avg_{key}", avg_time)
        
        # ─── Save Results Artifact ───────────────────────────────────────
        results_path = Path(__file__).parent / "results.json"
        save_results(results, scores, str(results_path))
        mlflow.log_artifact(str(results_path))
        
        # Log feedback stats if available
        feedback_stats = rag_engine.get_feedback_stats()
        if feedback_stats.get("total", 0) > 0:
            mlflow.log_metrics({
                "total_feedback": feedback_stats["total"],
                "thumbs_up": feedback_stats["thumbs_up"],
                "thumbs_down": feedback_stats["thumbs_down"],
                "satisfaction_rate": feedback_stats["satisfaction_rate"],
            })
        
        print("\n" + "=" * 60)
        print("✅ MLflow experiment logged successfully!")
        print(f"   Experiment: {experiment_name}")
        print(f"   View results: mlflow ui  (http://localhost:5000)")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run MLflow experiment tracking for RAG pipeline")
    parser.add_argument("--pdf", type=str, default="dummy.pdf", help="PDF to evaluate against")
    parser.add_argument("--session_id", type=str, help="Existing session ID")
    parser.add_argument("--experiment-name", type=str, default="RAG_Pipeline_Evaluation", help="MLflow experiment name")
    parser.add_argument("--dataset", type=str, help="Custom eval dataset JSON path")
    
    args = parser.parse_args()
    
    run_experiment(
        pdf_path=args.pdf,
        session_id=args.session_id,
        experiment_name=args.experiment_name,
        eval_dataset_path=args.dataset,
    )


if __name__ == "__main__":
    main()
