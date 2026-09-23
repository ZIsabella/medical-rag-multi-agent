import os
import re
import json
import math
import string
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Tuple
from collections import Counter

# Setting this env var ensures MLflow allows file-based registry
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import asyncpg
import httpx
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from bert_score import score
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluator")

# -------------------------
# Database configuration
# -------------------------
DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_NAME = os.getenv("POSTGRES_DB", "medical_rag")
DB_USER = os.getenv("POSTGRES_USER", "medical_user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "medical_password")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5432"))

# -------------------------
# Service Endpoints
# -------------------------
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator-service:8000/api/v1/ask")
RETRIEVAL_URL = os.getenv("RETRIEVAL_URL", "http://data-service:8001/api/v1/retrieval/search")

# -------------------------
# MLflow configuration
# -------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns")
MLFLOW_EXPERIMENT_NAME = "Medical-RAG-Evaluation"

OUTPUT_FILE = os.getenv("OUTPUT_FILE", "rag_evaluation_results.json")
SQL_LIMIT = int(os.getenv("SQL_LIMIT", "10"))
TOP_K = int(os.getenv("TOP_K", "5"))

# Model directories (Mount from host ./models to /app/models)
LOCAL_JUDGE_PATH = "/app/models/Qwen2-0.5B-Instruct"
LOCAL_EMBEDDING_PATH = "/app/models/all-MiniLM-L6-v2"

JUDGE_MODEL_ID = LOCAL_JUDGE_PATH if os.path.exists(LOCAL_JUDGE_PATH) else "Qwen/Qwen2-0.5B-Instruct"
EMBEDDING_MODEL_ID = LOCAL_EMBEDDING_PATH if os.path.exists(LOCAL_EMBEDDING_PATH) else "all-MiniLM-L6-v2"

device = "cpu"
logger.info("Targeting device: %s (Forced CPU evaluation)", device)

# Load Embedding Model
logger.info("Loading SentenceTransformer from: %s on %s...", EMBEDDING_MODEL_ID, device)
embed_model = SentenceTransformer(EMBEDDING_MODEL_ID, device=device)

# Load Judge LLM
logger.info("Loading Judge LLM from: %s on %s...", JUDGE_MODEL_ID, device)
judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_ID, local_files_only=True)

judge_config = AutoConfig.from_pretrained(JUDGE_MODEL_ID, local_files_only=True)
if getattr(judge_config, "rope_parameters", None) is None:
    judge_config.rope_parameters = {"rope_type": "default"}
elif isinstance(judge_config.rope_parameters, dict):
    judge_config.rope_parameters.setdefault("rope_type", "default")

judge_model = AutoModelForCausalLM.from_pretrained(
    JUDGE_MODEL_ID,
    config=judge_config,
    torch_dtype=torch.float32,
    device_map=None,
    low_cpu_mem_usage=True,
    local_files_only=True,
).to(device)
judge_model.eval()

# ROUGE Scorer
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(text.split())


def get_tokens(text: str) -> List[str]:
    return normalize_text(text).split()


def compute_token_precision_recall_f1(gold: str, pred: str) -> Tuple[float, float, float]:
    gold_toks = get_tokens(gold)
    pred_toks = get_tokens(pred)
    if not gold_toks or not pred_toks:
        return 0.0, 0.0, 0.0

    pred_counts = Counter(pred_toks)
    gold_counts = Counter(gold_toks)
    overlap = sum((pred_counts & gold_counts).values())
    if overlap == 0:
        return 0.0, 0.0, 0.0

    precision = overlap / len(pred_toks)
    recall = overlap / len(gold_toks)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_rouge_scores(pred: str, gold: str) -> Dict[str, float]:
    if not pred.strip() or not gold.strip():
        return {
            "rouge1_precision": 0.0, "rouge1_recall": 0.0, "rouge1_f1": 0.0,
            "rouge2_precision": 0.0, "rouge2_recall": 0.0, "rouge2_f1": 0.0,
            "rougeL_precision": 0.0, "rougeL_recall": 0.0, "rougeL_f1": 0.0,
        }
    scores = scorer.score(gold, pred)
    return {
        "rouge1_precision": scores["rouge1"].precision,
        "rouge1_recall": scores["rouge1"].recall,
        "rouge1_f1": scores["rouge1"].fmeasure,
        "rouge2_precision": scores["rouge2"].precision,
        "rouge2_recall": scores["rouge2"].recall,
        "rouge2_f1": scores["rouge2"].fmeasure,
        "rougeL_precision": scores["rougeL"].precision,
        "rougeL_recall": scores["rougeL"].recall,
        "rougeL_f1": scores["rougeL"].fmeasure,
    }


def compute_bertscore_batch(preds: List[str], golds: List[str]) -> Tuple[List[float], List[float], List[float]]:
    clean_preds = [p if p.strip() else "empty" for p in preds]
    clean_golds = [g if g.strip() else "empty" for g in golds]

    try:
        P, R, F1 = score(
            clean_preds,
            clean_golds,
            model_type="distilbert-base-uncased",
            lang="en",
            verbose=False,
            device=device,
        )
        p_list = [0.0 if not preds[i].strip() else float(P[i].item()) for i in range(len(preds))]
        r_list = [0.0 if not preds[i].strip() else float(R[i].item()) for i in range(len(preds))]
        f1_list = [0.0 if not preds[i].strip() else float(F1[i].item()) for i in range(len(preds))]
    except Exception as e:
        logger.warning("BERTScore offline calculation skipped or failed: %s", e)
        p_list = [0.0] * len(preds)
        r_list = [0.0] * len(preds)
        f1_list = [0.0] * len(preds)

    return p_list, r_list, f1_list


def compute_exact_match(gold: str, pred: str) -> float:
    return 1.0 if normalize_text(gold) == normalize_text(pred) else 0.0


def compute_jaccard(gold: str, pred: str) -> float:
    g_set = set(get_tokens(gold))
    p_set = set(get_tokens(pred))
    union = g_set | p_set
    if not union:
        return 0.0
    return len(g_set & p_set) / len(union)


def compute_semantic_similarity(gold: str, pred: str) -> float:
    if not gold.strip() or not pred.strip():
        return 0.0
    with torch.no_grad():
        emb_gold = embed_model.encode(gold, convert_to_tensor=True, device=device)
        emb_pred = embed_model.encode(pred, convert_to_tensor=True, device=device)
        sim = torch.cosine_similarity(emb_gold.unsqueeze(0), emb_pred.unsqueeze(0))
        return max(0.0, float(sim.item()))


def compute_length_stats(pred: str) -> Dict[str, Any]:
    tokens = get_tokens(pred)
    return {
        "char_length": len(pred),
        "word_count": len(tokens),
        "is_empty": len(tokens) == 0,
    }


def compute_dcg(relevances: List[int], k: int) -> float:
    dcg = 0.0
    for idx, rel in enumerate(relevances[:k]):
        dcg += (2 ** rel - 1) / math.log2(idx + 2)
    return dcg


def compute_ndcg(relevances: List[int], k: int) -> float:
    actual_dcg = compute_dcg(relevances, k)
    ideal_dcg = compute_dcg(sorted(relevances, reverse=True), k)
    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def _extract_scores(raw_text: str, has_context: bool) -> Tuple[float, float]:
    faithfulness = 0.0
    medical_accuracy = 0.0
    cleaned = raw_text.strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            faithfulness = float(data.get("faithfulness", 0.0))
            medical_accuracy = float(data.get("medical_accuracy", 0.0))
    except Exception:
        json_pattern = re.search(r"\{.*?\}", cleaned, re.DOTALL)
        if json_pattern:
            try:
                data = json.loads(json_pattern.group(0))
                if isinstance(data, dict):
                    faithfulness = float(data.get("faithfulness", 0.0))
                    medical_accuracy = float(data.get("medical_accuracy", 0.0))
            except Exception:
                pass

        if faithfulness == 0.0 and medical_accuracy == 0.0:
            f_match = re.search(r"(?:faithfulness|faith)\s*[:=\-]\s*([0-9]*\.?[0-9]+)", cleaned, re.IGNORECASE)
            m_match = re.search(r"(?:medical_accuracy|accuracy|medical)\s*[:=\-]\s*([0-9]*\.?[0-9]+)", cleaned,
                                re.IGNORECASE)
            if f_match:
                try:
                    faithfulness = float(f_match.group(1))
                except Exception:
                    pass
            if m_match:
                try:
                    medical_accuracy = float(m_match.group(1))
                except Exception:
                    pass

    if 1.0 < faithfulness <= 100.0:
        faithfulness /= 100.0
    if 1.0 < medical_accuracy <= 100.0:
        medical_accuracy /= 100.0

    if not has_context:
        f_final = 0.0
    else:
        f_final = max(0.0, min(1.0, faithfulness))
    m_final = max(0.0, min(1.0, medical_accuracy))

    return f_final, m_final


def get_llm_judgment(question: str, gold_answer: str, pred_answer: str, context: str) -> Tuple[float, float, str]:
    if not pred_answer.strip():
        return 0.0, 0.0, "Empty prediction response."

    has_context = bool(context and context.strip())
    context_str = context if has_context else "NO RETRIEVED CONTEXT AVAILABLE."

    prompt = f"""<|im_start|>system
You are an expert strict medical auditor evaluating a clinical RAG system.
Evaluate the PREDICTED ANSWER strictly on two dimensions:
1. "faithfulness": (0.0 to 1.0) Is the answer fully derived from the RETRIEVED CONTEXT without hallucinations? If no context is available, score 0.0.
2. "medical_accuracy": (0.0 to 1.0) Is the answer clinically factual, aligned with the GOLD ANSWER, and safe?
Output ONLY a valid JSON object matching this schema:
{{"faithfulness": 1.0, "medical_accuracy": 1.0}}
<|im_end|>
<|im_start|>user
[QUESTION]: {question}
[RETRIEVED CONTEXT]: {context_str}
[GOLD ANSWER]: {gold_answer}
[PREDICTED ANSWER]: {pred_answer}
<|im_end|>
<|im_start|>assistant
{{"faithfulness":"""

    inputs = judge_tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = judge_model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.01,
            do_sample=False,
            pad_token_id=judge_tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    output_text = '{"faithfulness":' + judge_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    f_score, m_score = _extract_scores(output_text, has_context=has_context)
    return f_score, m_score, output_text


class MedicalRAGEvaluator:
    def __init__(self):
        self.http_client = httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=15.0))

    async def close(self):
        try:
            await asyncio.wait_for(self.http_client.aclose(), timeout=2.0)
        except Exception:
            pass

    async def connect_db(self):
        logger.info("Connecting to database %s on %s:%d...", DB_NAME, DB_HOST, DB_PORT)
        return await asyncpg.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
        )

    async def fetch_gold_set(self) -> List[Dict[str, Any]]:
        conn = await self.connect_db()
        try:
            col_records = await conn.fetch(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'documents'"
            )
            cols = {r["column_name"] for r in col_records}

            id_col = "document_id" if "document_id" in cols else ("id" if "id" in cols else None)
            q_col = "question"
            a_col = "answer"

            if not id_col or q_col not in cols or a_col not in cols:
                logger.error("Required columns not found in 'documents'. Existing cols: %s", cols)
                return []

            query = f"""
                SELECT {id_col} AS doc_id, {q_col} AS question, {a_col} AS gold_answer
                FROM documents
                WHERE {q_col} IS NOT NULL AND TRIM({q_col}) <> ''
            """
            if SQL_LIMIT > 0:
                query += f" LIMIT {SQL_LIMIT}"

            rows = await conn.fetch(query)
            logger.info("Fetched %d gold question-answer pairs from database.", len(rows))
            return [
                {
                    "id": str(r["doc_id"]),
                    "question": r["question"].strip(),
                    "gold_answer": (r["gold_answer"] or "").strip(),
                }
                for r in rows
            ]
        finally:
            await conn.close()

    async def call_orchestrator(self, query: str) -> Tuple[str, str]:
        payload = {"query": query}
        try:
            response = await self.http_client.post(
                ORCHESTRATOR_URL,
                json=payload,
                headers={"Accept": "application/json, text/event-stream"},
            )
            response.raise_for_status()

            content_type = (response.headers.get("content-type", "") or "").lower()
            if "text/event-stream" in content_type:
                answer_chunks = []
                context_chunks = []

                for line in response.text.splitlines():
                    line = line.rstrip("\r\n")
                    if not line.startswith("data:"):
                        continue

                    raw_data = line[6:].strip() if line.startswith("data: ") else line[5:].strip()
                    if not raw_data or raw_data in ["[DONE]", "DONE"]:
                        continue

                    try:
                        chunk_json = json.loads(raw_data)
                    except Exception:
                        answer_chunks.append(raw_data)
                        continue

                    if isinstance(chunk_json, dict):
                        token_str = (
                                chunk_json.get("content")
                                or chunk_json.get("token")
                                or chunk_json.get("text")
                                or chunk_json.get("answer")
                        )
                        if token_str and isinstance(token_str, str):
                            answer_chunks.append(token_str)

                        ctx = (
                                chunk_json.get("context")
                                or chunk_json.get("retrieved_context")
                                or chunk_json.get("sources")
                                or chunk_json.get("documents")
                        )
                        if ctx:
                            if isinstance(ctx, list):
                                for item in ctx:
                                    if isinstance(item, dict):
                                        context_chunks.append(
                                            item.get("content") or item.get("text") or item.get("chunk") or str(item)
                                        )
                                    else:
                                        context_chunks.append(str(item))
                            else:
                                context_chunks.append(str(ctx))

                generated_answer = "".join(answer_chunks).strip()
                retrieved_context = "\n---\n".join(context_chunks).strip()
                return generated_answer, retrieved_context

            data = response.json()
            generated_answer = (
                    data.get("answer")
                    or data.get("generated_answer")
                    or data.get("response")
                    or data.get("content")
                    or ""
            ).strip()
            retrieved_context = str(data.get("context") or data.get("retrieved_context") or "").strip()
            return generated_answer, retrieved_context

        except Exception as e:
            logger.error("Error calling orchestrator: %s", e)
            return "", ""

    async def call_retrieval(self, query: str) -> List[Dict[str, Any]]:
        payload = {"query": query, "top_k": TOP_K}
        try:
            res = await self.http_client.post(RETRIEVAL_URL, json=payload)
            if res.status_code == 200:
                data = res.json()
                return data.get("results") or data.get("documents") or []
        except Exception as exc:
            logger.error("Retrieval call failed: %s", exc)
        return []

    async def run(self):
        # MLflow setup
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info("MLflow Tracking URI: %s", MLFLOW_TRACKING_URI)

        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        gold_pairs = await self.fetch_gold_set()
        if not gold_pairs:
            logger.error("No gold evaluation records found. Aborting.")
            return

        with mlflow.start_run(run_name="Medical_RAG_EndToEnd_Eval") as run:
            run_id = run.info.run_id
            logger.info("Started MLflow Run ID: %s", run_id)

            mlflow.log_params({
                "sample_size": len(gold_pairs),
                "top_k": TOP_K,
                "retrieval_url": RETRIEVAL_URL,
                "orchestrator_url": ORCHESTRATOR_URL,
                "judge_model": JUDGE_MODEL_ID,
                "embedding_model": EMBEDDING_MODEL_ID,
                "device": device,
            })

            preds = []
            golds = []
            results = []

            retrieval_hits = 0
            mrr_list = []
            ndcg_list = []

            for i, item in enumerate(gold_pairs, 1):
                doc_id = item["id"]
                q = item["question"]
                gold = item["gold_answer"]

                logger.info("[%d/%d] Querying: %s", i, len(gold_pairs), q[:60])

                # 1) Orchestrator call
                pred_answer, context = await self.call_orchestrator(q)

                # 2) Retrieval evaluation
                retrieved_docs = await self.call_retrieval(q)
                relevances = []
                rank_hit = 0

                for rank_idx, doc in enumerate(retrieved_docs, start=1):
                    ret_id = str(doc.get("document_id") or doc.get("id") or "")
                    if ret_id == doc_id:
                        relevances.append(1)
                        if rank_hit == 0:
                            rank_hit = rank_idx
                    else:
                        relevances.append(0)

                is_hit = 1 if rank_hit > 0 else 0
                retrieval_hits += is_hit
                mrr = (1.0 / rank_hit) if rank_hit > 0 else 0.0
                mrr_list.append(mrr)
                ndcg = compute_ndcg(relevances, TOP_K)
                ndcg_list.append(ndcg)

                # 3) Text metrics
                preds.append(pred_answer)
                golds.append(gold)

                prec, rec, f1 = compute_token_precision_recall_f1(gold, pred_answer)
                em = compute_exact_match(gold, pred_answer)
                jacc = compute_jaccard(gold, pred_answer)
                sem_sim = compute_semantic_similarity(gold, pred_answer)
                len_stats = compute_length_stats(pred_answer)
                rouge_res = compute_rouge_scores(pred_answer, gold)

                # 4) LLM judge
                faithfulness, medical_acc, raw_judge = get_llm_judgment(q, gold, pred_answer, context)

                result_entry = {
                    "doc_id": doc_id,
                    "question": q,
                    "gold_answer": gold,
                    "predicted_answer": pred_answer,
                    "retrieved_context": context,
                    "retrieval_hit": bool(is_hit),
                    "retrieval_mrr": mrr,
                    "retrieval_ndcg": ndcg,
                    "token_precision": prec,
                    "token_recall": rec,
                    "token_f1": f1,
                    "exact_match": em,
                    "jaccard_similarity": jacc,
                    "semantic_similarity": sem_sim,
                    "rouge": rouge_res,
                    "faithfulness": faithfulness,
                    "medical_accuracy": medical_acc,
                    "length_stats": len_stats,
                    "raw_judge_output": raw_judge,
                }
                results.append(result_entry)

            # 5) Batch BERTScore
            logger.info("Computing BERTScore in batch mode...")
            bert_p, bert_r, bert_f1 = compute_bertscore_batch(preds, golds)
            for i, entry in enumerate(results):
                entry["bertscore"] = {
                    "precision": bert_p[i],
                    "recall": bert_r[i],
                    "f1": bert_f1[i],
                }

            total = len(results)
            mean_metrics = {
                "retrieval_recall_at_k": retrieval_hits / total if total > 0 else 0.0,
                "retrieval_mean_mrr": sum(mrr_list) / total if total > 0 else 0.0,
                "retrieval_mean_ndcg": sum(ndcg_list) / total if total > 0 else 0.0,
                "mean_token_f1": sum(r["token_f1"] for r in results) / total if total > 0 else 0.0,
                "mean_token_precision": sum(r["token_precision"] for r in results) / total if total > 0 else 0.0,
                "mean_token_recall": sum(r["token_recall"] for r in results) / total if total > 0 else 0.0,
                "mean_exact_match": sum(r["exact_match"] for r in results) / total if total > 0 else 0.0,
                "mean_jaccard": sum(r["jaccard_similarity"] for r in results) / total if total > 0 else 0.0,
                "mean_semantic_similarity": sum(
                    r["semantic_similarity"] for r in results) / total if total > 0 else 0.0,
                "mean_rouge1_f1": sum(r["rouge"]["rouge1_f1"] for r in results) / total if total > 0 else 0.0,
                "mean_rouge1_precision": sum(
                    r["rouge"]["rouge1_precision"] for r in results) / total if total > 0 else 0.0,
                "mean_rouge1_recall": sum(r["rouge"]["rouge1_recall"] for r in results) / total if total > 0 else 0.0,
                "mean_rouge2_f1": sum(r["rouge"]["rouge2_f1"] for r in results) / total if total > 0 else 0.0,
                "mean_rougeL_f1": sum(r["rouge"]["rougeL_f1"] for r in results) / total if total > 0 else 0.0,
                "mean_bertscore_f1": sum(r["bertscore"]["f1"] for r in results) / total if total > 0 else 0.0,
                "mean_bertscore_precision": sum(
                    r["bertscore"]["precision"] for r in results) / total if total > 0 else 0.0,
                "mean_bertscore_recall": sum(r["bertscore"]["recall"] for r in results) / total if total > 0 else 0.0,
                "mean_faithfulness": sum(r["faithfulness"] for r in results) / total if total > 0 else 0.0,
                "mean_medical_accuracy": sum(r["medical_accuracy"] for r in results) / total if total > 0 else 0.0,
            }

            mlflow.log_metrics(mean_metrics)

            summary = {
                "timestamp": datetime.utcnow().isoformat(),
                "total_samples": total,
                "mean_metrics": mean_metrics,
                "results": results,
            }

            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            mlflow.log_artifact(OUTPUT_FILE)

            logger.info("Evaluation complete! Results saved to %s", OUTPUT_FILE)
            logger.info("All metrics & artifacts successfully logged to MLflow! Run ID: %s", run_id)


async def main():
    evaluator = MedicalRAGEvaluator()
    try:
        await evaluator.run()
    finally:
        await evaluator.close()


if __name__ == "__main__":
    asyncio.run(main())
