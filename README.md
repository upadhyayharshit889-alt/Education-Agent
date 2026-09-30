# 🎓 Education Agent — AI-Powered Personalized Learning Platform

An agentic AI learning assistant built with **RAG (Retrieval-Augmented Generation)**, **LangChain**, and **Ollama (LLaMA 3.2 + nomic-embed-text)**. It dynamically evaluates student academic performance, pinpoints subject bottlenecks, cross-checks syllabus coverage against student notes, and synthesizes personalized study blueprints.

---

## 🌟 Features

- **📊 Subject Performance Matrix**: Dynamic analytics across all course subjects with risk classification (`Needs Attention`, `Needs Improvement`, `Proficient`, `Good Performance`).
- **📋 Syllabus & Notes Gap Tracker**: Automatically detects curriculum topics missing from student notes.
- **🧠 Local RAG Pipeline**:
  - Embedding retrieval using `nomic-embed-text` with cosine similarity search.
  - Generative reasoning with `llama3.2` locally via Ollama (zero API costs, 100% private).
- **🗓️ Dynamic Study Planner**: Generates actionable 7-day day-by-day schedules with specific hourly routines and practice questions.
- **✨ RAG Transparency Inspector**: Shows the exact retrieved chunk from `student_data.txt` and the vector similarity percentage.
- **🌓 Dual Theme**: Ultra-modern glassmorphic interface with Dark and Light mode toggles.
- **⚡ Live Data Editor**: Edit student marks or curriculum directly from the browser without restarting the server.

---

## 🚀 How to Run the Website

### Method 1: One-Click Launcher (Windows)
Double-click [`run_website.bat`](run_website.bat). It will automatically start the server and open the website in your browser.

### Method 2: Command Line (PowerShell / Terminal)
```powershell
# 1. Navigate to the project directory
cd "c:\Users\DELL 7420 I7 11THGEN\Desktop\Agentic ai\Education-Agent"

# 2. Start the web server
$env:PYTHONIOENCODING="utf-8"
.\.venv\Scripts\python.exe server.py

# 3. Open in your browser:
# http://127.0.0.1:5000
```

---

## 📦 Prerequisites

1. **Python 3.10+** (Virtual environment included in `.venv/`)
2. **Ollama** running locally with the required models:
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

---

## 📂 Project Architecture

```
Education-Agent/
│
├── server.py              # Flask Web Backend with RAG & Ollama endpoints
├── templates/
│   └── index.html         # Modern web application UI
├── static/
│   ├── css/
│   │   └── style.css      # Custom glassmorphism & responsive stylesheet
│   └── js/
│       └── app.js         # Interactive controller & dynamic RAG visualizer
│
├── student_data.txt       # Active student marks and syllabus data
├── run_website.bat        # 1-click Windows launcher
├── requirements.txt       # Python package dependencies
├── .gitignore             # Git ignore configuration
│
└── app_final.py           # Original Tkinter desktop application
```

---

## 🐙 How to Upload to GitHub

### Option A: Using Git (Recommended)

1. **Install Git** if not already installed: [https://git-scm.com/download/win](https://git-scm.com/download/win)
2. Open PowerShell in this folder and run:
   ```powershell
   git init
   git add .
   git commit -m "Initial commit: Education Agent web platform with RAG & Ollama"
   git branch -M main
   git remote add origin https://github.com/<YOUR-USERNAME>/<YOUR-REPOSITORY-NAME>.git
   git push -u origin main
   ```

### Option B: Upload via GitHub.com (No Git Installation Required)

1. Go to [https://github.com/new](https://github.com/new) and create a new repository (e.g. `Education-Agent`).
2. On the new repository page, click **"uploading an existing file"**.
3. Drag and drop all project files **except** `.venv/` into the browser.
4. Click **"Commit changes"**.
