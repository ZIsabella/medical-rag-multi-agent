🏥 Medical RAG Multi-Agent System
An intelligent, distributed system for answering medical queries based on RAG (Retrieval-Augmented Generation) and a Multi-Agent architecture. This project leverages specialized medical language models (like BioMistral) and advanced embedding models to perform semantic searches on medical datasets (MedQuAD and PubMedQA).
 
🛠 Architecture and Microservices
The system consists of 8 distinct containers within the medical-rag-net dedicated network:


Service	Container Name	Host:Container Port	Description / Role
Orchestrator	medical_orchestrator	8000:8000	Data flow management, routing requests between agents
Data Service	medical_data	8001:8001	Embedding and semantic retrieval service (Vector Search)
LLM Service	medical_llm	8002:8002	Language model inference service (GGUF / Ollama)
Frontend	medical_frontend	3000:3000	Chatbot UI based on Next.js
PostgreSQL	medical_postgres	5432:5432	Main database with pgvector extension
Redis	medical_redis	6379:6379	Caching system for fast response times
Ingest Worker	medical_ingest	Manual Profile	Processing, chunking, and indexing medical datasets
Stress Eval	medical_evaluator	Manual Profile	Stress testing and system efficiency evaluation with MLflow
Metrics Eval	metrics_eval	Manual Profile	Calculating and logging RAG evaluation metrics

💻 Prerequisites
Docker Desktop (with Docker Compose v2)
Windows OS (must run in CMD environment)
At least 16GB RAM (32GB recommended for BioMistral model)

🚀 Quick Start
1. Clone the repository
cmd
git clone https://github.com/your-username/medical-rag-multi-agent.git
cd medical-rag-multi-agent

3. Build and Start All Core Services
To build and start the core project services:

cmd
docker compose up -d --build
🐳 Docker Commands by Microservice
Note: All commands below are configured to run in Windows CMD.

1️⃣ Database Service (PostgreSQL + pgvector)
cmd
:: Start database
docker compose up -d postgres

:: View database logs
docker compose logs -f postgres

2️⃣ Caching Service (Redis)
cmd
:: Start Redis
docker compose up -d redis

3️⃣ Data and Embedding Service (Data Service)
cmd
:: Build and start Data Service
docker compose up -d --build data-service

:: View Data Service logs
docker compose logs -f data-service

4️⃣ Language Model Service (LLM Service)
cmd
:: Build and start LLM Service
docker compose up -d --build llm-service

:: View model loading logs
docker compose logs -f llm-service

5️⃣ Orchestrator Service
cmd
:: Build and start Orchestrator Service
docker compose up -d --build orchestrator-service

:: View logs
docker compose logs -f orchestrator-service

6️⃣ Frontend
cmd
:: Build and start Frontend
docker compose up -d --build frontend

:: After execution, open your browser at:
:: http://localhost:3000

📥 Data Ingestion
To process local datasets (MedQuAD and PubMedQA) and import them into the vector database:

cmd
:: Execute in Manual profile mode with mount connected to raw data folder
docker compose up  ingest-worker 
or
docker compose up -d ingest-worker
	
🧪 Stress and Metrics Evaluation
Run Stress Test with MLflow:
cmd
docker compose up stress-eval
or
docker compose up -d stress-eval

:: View progress and logs
docker compose logs -f stress-eval

Run RAG Metrics Calculation:
cmd
docker compose up metrics-eval
or
docker compose up -d -eval

⚙️ Project Management Commands

Operation	Command in CMD
Check container status	docker compose ps
View logs for all services	docker compose logs -f
Stop the whole system	docker compose down
Stop and remove volumes	docker compose down -v
Rebuild a specific service	docker compose build --no-cache <service_name>
📡 Main API Endpoints
Orchestrator Chat: POST http://localhost:8000/api/v1/chat
Data Retrieval Search: POST http://localhost:8001/api/v1/retrieval/search
LLM Health Check: GET http://localhost:8002/health
Frontend Web UI: http://localhost:3000
Mlflow Web UI: http://localhost:3000

