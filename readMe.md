# Backend Setup & Run Guide

## 1. Prerequisites

- **Python 3.8+** (preferably Python 3.10 or newer)
- **pip** (Python package manager)
- **MongoDB** (if the backend uses a local MongoDB instance)
- (Optional) **virtualenv** for isolated Python environments
- (Optional) **Docker** if you want to run the backend in a container

---

## 2. Setup Instructions

### a. Clone the Repository

```sh
git clone <your-repo-url>
cd phamiq/backend
```

### b. Create and Activate a Virtual Environment (Recommended)

```sh
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### c. Install Python Dependencies

```sh
pip install --upgrade pip
pip install -r requirements.txt
```

### d. Set Up Environment Variables

Check if there is a `.env` file required (not present in the snapshot, but often needed). If not, check `app/config.py` for any required environment variables (like database URIs, secret keys, etc.). Create a `.env` file if needed.

Example `.env` (edit as needed):

```
MONGO_URI=mongodb://localhost:27017/phamiq
SECRET_KEY=your_secret_key
```

### e. Start MongoDB

If you are running MongoDB locally, make sure it is running:

```sh
# On Windows, use MongoDB Compass or run:
mongod
```

### f. Run the Backend Server

You can start the backend server using the provided entry point. There are two common ways:

#### Option 1: Using `run.py`

```sh
python run.py
```

#### Option 2: Using `uvicorn` (if FastAPI is used)

If the backend uses FastAPI (likely, given the structure), you can run:

```sh
uvicorn app.main:app --reload
```

- `--reload` enables auto-reload on code changes (for development).
- The default port is 8000. You can specify another port with `--port 8080`.

### g. (Optional) Run with Docker

If you prefer Docker, and a `Dockerfile` is present:

```sh
docker build -t phamiq-backend .
docker run -p 8000:8000 phamiq-backend
```

---

## 3. Accessing the API

- By default, the API will be available at:  
  `http://localhost:8000`
- If using FastAPI, interactive docs are at:  
  `http://localhost:8000/docs`

---

## 4. Project Structure (Backend)

- `app/` – Main backend application
  - `main.py` – Main entry point (FastAPI/Flask app)
  - `config.py` – Configuration (env variables, settings)
  - `db/` – Database connection logic
  - `models/` – Data models and schemas
  - `routes/` – API route definitions
  - `services/` – Business logic/services
- `run.py` – Alternate entry point to start the server
- `requirements.txt` – Python dependencies
- `Dockerfile` – For containerized deployment

---

## 5. Troubleshooting

- If you get a "ModuleNotFoundError", ensure your virtual environment is activated and dependencies are installed.
- If MongoDB connection fails, check your `MONGO_URI` and ensure MongoDB is running.
- For CORS or network issues, check FastAPI/Flask CORS settings in `main.py`.
