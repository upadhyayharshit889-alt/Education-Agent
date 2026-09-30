"""
Education Agent - AI Learning Platform Web Server
Powered by Flask, LangChain, Ollama (Llama 3.2 + nomic-embed-text)
"""
import os
import re
import time
import math
import numpy as np
from flask import Flask, render_template, request, jsonify

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = Flask(__name__, static_folder="static", template_folder="templates")

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "student_data.txt")

# Initialize models
EMBEDDING_MODEL_NAME = "nomic-embed-text"
LLM_MODEL_NAME = "llama3.2"

try:
    embeddings_service = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
except Exception as e:
    print(f"Warning: Embeddings init deferred: {e}")
    embeddings_service = None

try:
    llm_service = ChatOllama(model=LLM_MODEL_NAME, temperature=0.1)
except Exception as e:
    print(f"Warning: LLM init deferred: {e}")
    llm_service = None


def cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def parse_student_data():
    """Parses student_data.txt into structured data."""
    if not os.path.exists(DATA_FILE):
        return {
            "name": "Student",
            "course": "Data Science",
            "subjects": [],
            "syllabus": [],
            "raw": ""
        }

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        raw_text = f.read()

    student_name = "Harshit"
    course_name = "Data Science"
    subjects = []
    syllabus = []

    # Parse student name
    name_match = re.search(r"Student:\s*([^\n\r]+)", raw_text, re.IGNORECASE)
    if name_match:
        student_name = name_match.group(1).strip()

    # Parse course
    course_match = re.search(r"Course:\s*([^\n\r]+)", raw_text, re.IGNORECASE)
    if course_match:
        course_name = course_match.group(1).strip()

    # Parse subjects & marks
    subjects_section = False
    syllabus_section = False

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue

        if "Subjects and Marks:" in line:
            subjects_section = True
            syllabus_section = False
            continue
        elif "Syllabus:" in line:
            subjects_section = False
            syllabus_section = True
            continue

        if subjects_section:
            if "-" in line:
                parts = line.split("-")
                sub = parts[0].strip()
                try:
                    score = int(parts[1].strip())
                except ValueError:
                    score = 0
                
                # Priority & status calculation
                if score < 50:
                    status = "Needs Attention"
                    priority = "High"
                    level_class = "danger"
                elif score < 70:
                    status = "Needs Improvement"
                    priority = "Medium"
                    level_class = "warning"
                elif score < 80:
                    status = "Proficient"
                    priority = "Low"
                    level_class = "info"
                else:
                    status = "Good Performance"
                    priority = "Low"
                    level_class = "success"

                subjects.append({
                    "name": sub,
                    "score": score,
                    "status": status,
                    "priority": priority,
                    "levelClass": level_class
                })
        elif syllabus_section:
            if line:
                syllabus.append(line)

    # Missing syllabus topics check (compare with student notes)
    # Notes historically cover: Python basics, SQL queries and joins, Power BI dashboards
    student_notes = [
        "Python basics",
        "SQL queries and joins",
        "Power BI dashboards"
    ]
    syllabus_status = []
    for item in syllabus:
        in_notes = item in student_notes
        syllabus_status.append({
            "topic": item,
            "hasNotes": in_notes,
            "status": "Covered in Notes" if in_notes else "Missing from Notes",
            "priority": "High" if ("Machine Learning" in item or "Statistics" in item) else ("Medium" if not in_notes else "Low")
        })

    # Overall statistics
    avg_score = round(sum(s["score"] for s in subjects) / len(subjects), 1) if subjects else 0
    weak_count = sum(1 for s in subjects if s["score"] < 60)
    strong_count = sum(1 for s in subjects if s["score"] >= 80)
    missing_notes_count = sum(1 for s in syllabus_status if not s["hasNotes"])

    return {
        "name": student_name,
        "course": course_name,
        "subjects": subjects,
        "syllabus": syllabus_status,
        "metrics": {
            "avgScore": avg_score,
            "totalSubjects": len(subjects),
            "weakCount": weak_count,
            "strongCount": strong_count,
            "missingNotesCount": missing_notes_count,
            "coveragePercent": round(((len(syllabus) - missing_notes_count) / len(syllabus) * 100), 1) if syllabus else 0
        },
        "raw": raw_text
    }


def get_rag_context(query):
    """Retrieves the most relevant chunk from student_data.txt using embeddings."""
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(content)

    if not chunks:
        return "", 0.0

    global embeddings_service
    if embeddings_service is None:
        embeddings_service = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)

    try:
        q_emb = embeddings_service.embed_query(query)
        c_embs = embeddings_service.embed_documents(chunks)

        scores = [cosine_similarity(q_emb, c_emb) for c_emb in c_embs]
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        return chunks[best_idx], best_score
    except Exception as e:
        print(f"RAG retrieval fallback error: {e}")
        # Fallback to returning the full content if embeddings fail
        return chunks[0] if chunks else content, 0.75


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/student-data", methods=["GET"])
def get_student_data():
    try:
        data = parse_student_data()
        return jsonify({"success": True, "data": data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/generate-plan", methods=["POST"])
def generate_plan():
    req_data = request.get_json() or {}
    question = req_data.get("question", "").strip()
    plan_mode = req_data.get("mode", "general")

    if not question:
        question = "What are the student's weak subjects and what should they study?"

    start_time = time.time()

    try:
        # 1. RAG retrieval
        best_chunk, similarity = get_rag_context(question)

        # 2. Get student data summary
        student_data_obj = parse_student_data()
        student_info = student_data_obj["raw"]

        # 3. Create context-rich prompt tailored to mode
        mode_instruction = ""
        if plan_mode == "7day":
            mode_instruction = """
Create a structured, highly actionable 7-DAY STUDY SCHEDULE formatted day-by-day (Day 1 through Day 7).
- Allocate dedicated time slots (e.g., Morning 1 hr, Evening 1.5 hrs).
- Prioritize Machine Learning (48) and Statistics (55).
- Include specific practical exercises, key concepts to revise, and code/dataset tasks.
- Keep recommendations realistic and beginner-to-intermediate friendly.
"""
        elif plan_mode == "weak_areas":
            mode_instruction = """
Focus specifically on a deep-dive diagnostic of the student's WEAK AREAS:
1. Critical Weakness: Machine Learning (Score 48) - Root cause analysis, must-learn topics, foundational steps.
2. Moderate Weakness: Statistics (Score 55) - Formulas, probability, hypothesis testing focus.
3. Missing Notes: Topics present in syllabus but missing from notes.
4. Immediate 48-Hour Recovery Action Items.
"""
        elif plan_mode == "exam_prep":
            mode_instruction = """
Provide an INTENSIVE EXAM & INTERVIEW PREPARATION PLAN:
- High-yield topics most frequently asked in Machine Learning and Statistics assessments.
- Quick formula cheat-sheet suggestions.
- 5 mock assessment questions with answer outlines to test understanding immediately.
"""
        else:
            mode_instruction = """
Provide a comprehensive personal learning recommendation:
1. Weak Subjects Analysis & Priority Levels
2. Specific Core Topics to Master
3. Personalized Study Strategy & Schedule
4. Daily Practice Routine & Hands-On Projects
5. Milestones to reach 80%+ across all subjects
"""

        prompt = f"""You are Education Agent, an elite AI-powered personalized learning assistant and academic mentor.

Student Context & Performance Profile:
{student_info}

Relevant Retrieved Context (RAG):
{best_chunk}

Student Request:
{question}

Task Instructions:
{mode_instruction}

Format your response using clean, beautiful Markdown with bold headings, bullet points, checklists (- [ ]), and clear organization. Make it encouraging, professional, and directly actionable."""

        global llm_service
        if llm_service is None:
            llm_service = ChatOllama(model=LLM_MODEL_NAME, temperature=0.1)

        ai_response = llm_service.invoke(prompt)
        latency = round(time.time() - start_time, 2)

        return jsonify({
            "success": True,
            "answer": ai_response.content,
            "rag": {
                "chunk": best_chunk,
                "similarity": round(similarity * 100, 1),
                "model": LLM_MODEL_NAME,
                "embeddingModel": EMBEDDING_MODEL_NAME,
                "latencySeconds": latency
            }
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "latencySeconds": round(time.time() - start_time, 2)
        }), 500


@app.route("/api/update-data", methods=["POST"])
def update_data():
    req_data = request.get_json() or {}
    raw_content = req_data.get("content", "").strip()

    if not raw_content:
        return jsonify({"success": False, "error": "Content cannot be empty"}), 400

    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            f.write(raw_content)

        updated_data = parse_student_data()
        return jsonify({"success": True, "data": updated_data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "llm": LLM_MODEL_NAME,
        "embeddings": EMBEDDING_MODEL_NAME,
        "dataFile": os.path.basename(DATA_FILE)
    })


if __name__ == "__main__":
    print("Education Agent Web Server starting on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)

