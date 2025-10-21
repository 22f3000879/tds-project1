Tds project 1 

## Project: LLM Code Deployment

This project implements a fully autonomous API endpoint designed to receive a task brief, use a Large Language Model (LLM) to generate a functional web application, deploy it to GitHub Pages, and notify an external evaluation service. It supports both initial deployment (Round 1) and feature revision (Round 2).

The application is deployed on **Render** to ensure a persistent, publicly accessible service as required for evaluation.

---

## 🚀 Live Demo and Deployment

| Component | Status | Details |
| :--- | :--- | :--- |
| **Live API Endpoint** | **Active** | `https://tds-project1-8hnp.onrender.com/' |
| **Deployment User** | `22f3000879` | All generated projects are deployed under this GitHub account. |

### Key Features
* **Asynchronous Processing:** Handles task requests instantly (HTTP 200) and executes long-running deployment tasks in the background using FastAPI and `asyncio`.
* **Secure Configuration:** Uses environment variables (secured via Render Environment Variables) for all sensitive credentials (`GITHUB_TOKEN`, `OPENAI_API_KEY`, `STUDENT_SECRET`).
* **LLM Tooling:** Employs OpenAI's function calling feature to force the LLM to output precise, validated JSON structures containing `index.html`, `README.md`, and `LICENSE`.
* **Robust Deployment:** Automates repository creation, Git committing, and GitHub Pages activation/update using `GitPython` and the GitHub REST API.
* **Resilient Notification:** Implements **exponential backoff** for retrying the final notification to the instructor's `evaluation_url`.

---

## ⚙️ Setup and Usage

### 1. Local Development Setup

To run this application locally (for testing before deployment):

1.  **Clone the Repository:**
    ```bash
    git clone [Your Space Git URL]
    cd llm_Deployement
    ```

2.  **Install Dependencies:** Ensure you have Git installed, then install the Python libraries:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Secrets:** Create a local `.env` file and fill in your actual credentials:
    ```env
    # .env
    STUDENT_SECRET="YOUR_UNIQUE_SECRET_FOR_EVALUATOR_VERIFICATION"
    GITHUB_TOKEN="ghp_..."
    GITHUB_USERNAME="22f3000879"
    OPENAI_API_KEY="sk-proj-..."
    # ... other config variables ...
    ```

4.  **Run Locally:**
    ```bash
    python -m uvicorn api_app:app --host 0.0.0.0 --port 8000 --reload
    ```
    Access the app at `http://localhost:8000/`.

### 2. Testing with a Sample Request (Evaluation Simulation)

Use `curl` or Postman to simulate the instructor's submission and trigger a live deployment:

```bash
curl -X POST "http://localhost:8000/ready" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@student.edu",
    "secret": "YOUR_UNIQUE_SECRET_FOR_EVALUATOR_VERIFICATION",
    "task": "test-task-sum-of-sales-123",
    "round": 1,
    "nonce": "test-nonce-abc",
    "brief": "Publish a simple Bootstrap page that sums sales from attached data.csv.",
    "checks": ["Repo has MIT license", "Page displays the correct total sales"],
    "evaluation_url": "[https://example.com/notify](https://example.com/notify)"
}'
