# G-14_Nest_Nitraipur

# Neural Clinical Data Mesh (NCDM)

Strategic Framework for Real-Time Clinical Dataflow Optimization using Agentic AI and Graph-Based Data Architecture.

This repository contains the complete source code for the NCDM platform, including the Python FastAPI backend and the React/Vite frontend.

## Prerequisites

Before setting up the project, ensure you have the following installed:

- **Python**: 3.10 or higher
- **Node.js**: v18 or higher (LTS recommended)
- **Git**: For version control

## Project Structure

- `clinical_dataflow_optimizer/api` - Backend REST API (FastAPI)
- `clinical_dataflow_optimizer/frontend` - Frontend Web Application (React + Vite)
- `clinical_dataflow_optimizer` - Core logic, Analysis pipeline, and AI Agents

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/mahes-reddy332/G-14_Nest_Nitraipur.git
cd G-14_Nest_Nitraipur
```

### 2. Backend Setup

The backend handles the API, database connectivity, and AI agents.

1. Navigate to the backend directory:
   ```bash
   cd clinical_dataflow_optimizer/api
   ```

2. Create a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: deeper dependencies might be listed in `../requirements-web.txt` or `../setup.py`. If you encounter missing packages, try installing from the parent directory as well.)*

4. Start the Backend Server:
   ```bash
   uvicorn main:app --reload
   ```
   The backend will start at `http://127.0.0.1:8000`.
   - **API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - **Health Check**: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

### 3. Frontend Setup

The frontend is a React application built with Vite.

1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd clinical_dataflow_optimizer/frontend
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

3. **IMPORTANT: API Configuration**
   Ensure your environment configuration points to the correct backend port (8000).
   Check `.env.development`:
   ```properties
   # Correct Configuration
   VITE_API_BASE_URL=/api
   VITE_WS_BASE_URL=
   ```
   *If you see `http://localhost:5000`, please change it to `/api` to use the proxy.*

4. Start the Frontend Development Server:
   ```bash
   npm run dev
   ```
   The frontend will typically start at `http://localhost:5173`.

## Running the Application

Once both servers are running:

1. Open your browser and navigate to **`http://localhost:5173`**
2. The dashboard should load and display real-time data from the backend.
3. If you see a "Starting..." status for a long time (more than 2 minutes), checking `http://localhost:8000/api/startup-status` will show the initialization progress.

## Troubleshooting

- **"WebSocket connection failed"**:
    - Ensure the backend is running on port 8000.
    - Check that `.env.development` in `frontend/` does **not** have `VITE_WS_BASE_URL=ws://localhost:5000`. It should be empty or point to 8000.
- **"No data available"**:
    - The backend runs a heavy data ingestion process on startup. Wait ~10-15 minutes for the initial full load, or check the logs for completion.
