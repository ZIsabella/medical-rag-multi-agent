import asyncio
import time
import httpx
import os
import mlflow


os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# ========================================
HOST = os.getenv("ORCHESTRATOR_HOST", "orchestrator-service")
PORT = os.getenv("ORCHESTRATOR_PORT", "8000")
API_URL = f"http://{HOST}:{PORT}/api/v1/chat"

API_TIMEOUT = 600.0

CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY_LIMIT", "1"))
TOTAL_REQUESTS = int(os.getenv("TOTAL_REQUESTS", "4"))

TEST_QUERIES = [
    "What are the symptoms of asthma?",
    "Can beta-blockers exacerbate symptoms in patients with severe asthma and hypertension?",
    "What are the common complications of type 2 diabetes?",
    "How does hypertension lead to chronic kidney disease?"
]

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Medical-RAG-Stress-Test")
# =================================================


async def send_request(client: httpx.AsyncClient, query: str, req_id: int) -> dict:
    start_time = time.perf_counter()
    try:
        full_text = ""
        status = None

        async with client.stream("POST", API_URL, json={"query": query}) as response:
            status = response.status_code
            async for chunk in response.aiter_text():
                full_text += chunk

        elapsed = time.perf_counter() - start_time

        is_success = (
            status == 200
            and len(full_text.strip()) > 0
            and "Error:" not in full_text
        )

        if is_success:
            print(f"[Req #{req_id:02d}] SUCCESS | Time: {elapsed:.2f}s | Len: {len(full_text)}")
        else:
            print(f"[Req #{req_id:02d}] FAILED  | Status: {status} | Time: {elapsed:.2f}s | Text: {full_text[:80]!r}")

        return {
            "req_id": req_id,
            "status": status,
            "time": elapsed,
            "success": is_success,
            "response_length": len(full_text)
        }

    except Exception as e:
        elapsed = time.perf_counter() - start_time
        err_msg = repr(e) if not str(e) else str(e)
        print(f"[Req #{req_id:02d}] EXCEPTION | Time: {elapsed:.2f}s | Error: {err_msg}")
        return {
            "req_id": req_id,
            "status": None,
            "time": elapsed,
            "success": False,
            "response_length": 0
        }


async def run_stress_test():
    print("=" * 60)
    print(f"Target URL          : {API_URL}")
    print(f"Total Requests      : {TOTAL_REQUESTS}")
    print(f"Concurrency Limit   : {CONCURRENCY_LIMIT}")
    print(f"Timeout per request : {API_TIMEOUT:.0f}s")
    print(f"MLflow Tracking URI : {MLFLOW_TRACKING_URI}")
    print(f"MLflow Experiment   : {MLFLOW_EXPERIMENT_NAME}")
    print("=" * 60)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def worker(client: httpx.AsyncClient, query: str, req_id: int) -> dict:
        async with semaphore:
            return await send_request(client, query, req_id)

    timeout = httpx.Timeout(API_TIMEOUT, connect=15.0, read=API_TIMEOUT, write=30.0, pool=API_TIMEOUT)

    with mlflow.start_run(run_name="Docker_Stress_Test_Run") as run:
        run_id = run.info.run_id
        print(f"Started MLflow Run ID: {run_id}")

        mlflow.log_params({
            "api_url": API_URL,
            "total_requests": TOTAL_REQUESTS,
            "concurrency_limit": CONCURRENCY_LIMIT,
            "timeout_seconds": API_TIMEOUT,
        })

        async with httpx.AsyncClient(timeout=timeout) as client:
            start_all = time.perf_counter()
            tasks = [
                worker(client, TEST_QUERIES[i % len(TEST_QUERIES)], i + 1)
                for i in range(TOTAL_REQUESTS)
            ]
            results = await asyncio.gather(*tasks)
            total_wall_time = time.perf_counter() - start_all

        successful = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]
        durations = [r["time"] for r in successful]

        avg_latency = sum(durations) / len(durations) if durations else 0.0
        min_latency = min(durations) if durations else 0.0
        max_latency = max(durations) if durations else 0.0
        throughput = (len(successful) / total_wall_time) if total_wall_time > 0 else 0.0

        metrics = {
            "total_wall_time": total_wall_time,
            "total_requests": TOTAL_REQUESTS,
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "success_rate": len(successful) / TOTAL_REQUESTS if TOTAL_REQUESTS > 0 else 0.0,
            "average_latency": avg_latency,
            "min_latency": min_latency,
            "max_latency": max_latency,
            "throughput_rps": throughput,
        }

        mlflow.log_metrics(metrics)

        print("\n" + "=" * 60)
        print("STRESS TEST SUMMARY & MLFLOW LOGGED")
        print("=" * 60)
        print(f"Total Wall Clock Time : {total_wall_time:.2f} s")
        print(f"Total Requests        : {TOTAL_REQUESTS}")
        print(f"Successful Requests   : {len(successful)}")
        print(f"Failed Requests       : {len(failed)}")
        if durations:
            print(f"Average Latency       : {avg_latency:.2f} s")
            print(f"Min Latency           : {min_latency:.2f} s")
            print(f"Max Latency           : {max_latency:.2f} s")
        print(f"Throughput (RPS)      : {throughput:.4f} req/s")
        if failed:
            print("\nFailed Request IDs    :", [r["req_id"] for r in failed])
        print(f"MLflow Run ID         : {run_id}")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_stress_test())
