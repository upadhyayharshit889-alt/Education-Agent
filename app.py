import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
import numpy as np

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================
# COLORS
# =========================
BG = "#F4F7FB"
CARD = "#FFFFFF"
TEXT = "#172033"
MUTED = "#667085"
PRIMARY = "#4F46E5"
SUCCESS = "#16A34A"
WARNING = "#F59E0B"
DANGER = "#DC2626"


# =========================
# LOAD STUDENT DATA
# =========================
def load_student_data():
    try:
        with open("student_data.txt", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Unable to load student data: {e}"


student_data = load_student_data()


# =========================
# CREATE CHUNKS
# =========================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(student_data)


# =========================
# EMBEDDINGS
# =========================
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# =========================
# COSINE SIMILARITY
# =========================
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0

    return np.dot(a, b) / denominator


# =========================
# RAG SEARCH
# =========================
def retrieve_best_chunk(question):

    question_embedding = embeddings.embed_query(question)

    chunk_embeddings = embeddings.embed_documents(chunks)

    scores = []

    for i, emb in enumerate(chunk_embeddings):
        score = cosine_similarity(question_embedding, emb)
        scores.append((score, i))

    scores.sort(reverse=True)

    best_index = scores[0][1]

    return chunks[best_index]


# =========================
# AI GENERATION
# =========================
def generate_ai_response(question):

    try:

        retrieved_context = retrieve_best_chunk(question)

        llm = ChatOllama(
            model="llama3.2",
            temperature=0
        )

        prompt = f"""
You are an AI Education Agent.

Analyze the student's learning performance and create a simple,
personalized study plan.

Student Information:
{student_data}

Relevant Retrieved Information:
{retrieved_context}

User Question:
{question}

Give the answer in a clear format:

1. Weak Subjects
2. Priority Level
3. Topics to Study
4. Personalized Study Plan
5. Recommended Daily Practice

Keep the explanation simple and practical for a student.
"""

        response = llm.invoke(prompt)

        return response.content

    except Exception as e:
        return f"Error while generating AI response:\n\n{str(e)}"


# =========================
# SHOW RESULT
# =========================
def show_result(result):

    result_box.config(state="normal")
    result_box.delete("1.0", tk.END)
    result_box.insert(tk.END, result)
    result_box.config(state="disabled")

    status_label.config(
        text="✓ Personalized study plan generated",
        foreground=SUCCESS
    )

    generate_button.config(
        state="normal",
        text="✨ Generate Study Plan"
    )


# =========================
# GENERATE BUTTON
# =========================
def generate_plan():

    question = question_box.get("1.0", tk.END).strip()

    if not question:
        messagebox.showwarning(
            "Question Required",
            "Please enter a question."
        )
        return

    generate_button.config(
        state="disabled",
        text="⏳ Generating..."
    )

    status_label.config(
        text="AI is analyzing your learning performance...",
        foreground=PRIMARY
    )

    result_box.config(state="normal")
    result_box.delete("1.0", tk.END)
    result_box.insert(
        tk.END,
        "🤖 AI is preparing your personalized study plan...\n\n"
        "Please wait..."
    )
    result_box.config(state="disabled")

    # Run AI in background
    thread = threading.Thread(
        target=run_ai,
        args=(question,),
        daemon=True
    )

    thread.start()


def run_ai(question):

    result = generate_ai_response(question)

    window.after(
        0,
        lambda: show_result(result)
    )


# =========================
# MAIN WINDOW
# =========================
window = tk.Tk()

window.title("Education Agent")

window.geometry("1100x850")

window.minsize(850, 650)

window.configure(bg=BG)


# =========================
# MAIN SCROLLABLE AREA
# =========================

outer_frame = tk.Frame(
    window,
    bg=BG
)

outer_frame.pack(
    fill="both",
    expand=True
)


canvas = tk.Canvas(
    outer_frame,
    bg=BG,
    highlightthickness=0
)

scrollbar = ttk.Scrollbar(
    outer_frame,
    orient="vertical",
    command=canvas.yview
)

canvas.configure(
    yscrollcommand=scrollbar.set
)

scrollbar.pack(
    side="right",
    fill="y"
)

canvas.pack(
    side="left",
    fill="both",
    expand=True
)


content_frame = tk.Frame(
    canvas,
    bg=BG
)

canvas_window = canvas.create_window(
    (0, 0),
    window=content_frame,
    anchor="nw"
)


def update_scroll_region(event=None):

    canvas.configure(
        scrollregion=canvas.bbox("all")
    )


content_frame.bind(
    "<Configure>",
    update_scroll_region
)


def resize_content(event):

    canvas.itemconfig(
        canvas_window,
        width=event.width
    )


canvas.bind(
    "<Configure>",
    resize_content
)


# Mouse wheel scrolling
def mouse_scroll(event):

    canvas.yview_scroll(
        int(-1 * (event.delta / 120)),
        "units"
    )


canvas.bind_all(
    "<MouseWheel>",
    mouse_scroll
)


# =========================
# HEADER
# =========================

header = tk.Frame(
    content_frame,
    bg=PRIMARY,
    height=110
)

header.pack(
    fill="x"
)

tk.Label(
    header,
    text="🎓 EDUCATION AGENT",
    font=("Segoe UI", 25, "bold"),
    bg=PRIMARY,
    fg="white"
).pack(
    pady=(20, 3)
)

tk.Label(
    header,
    text="AI-Powered Personalized Learning Assistant",
    font=("Segoe UI", 11),
    bg=PRIMARY,
    fg="white"
).pack()


# =========================
# STUDENT PROFILE CARD
# =========================

profile_card = tk.Frame(
    content_frame,
    bg=CARD,
    padx=25,
    pady=18
)

profile_card.pack(
    fill="x",
    padx=25,
    pady=(20, 10)
)

tk.Label(
    profile_card,
    text="👤  Student Profile",
    font=("Segoe UI", 16, "bold"),
    bg=CARD,
    fg=TEXT
).pack(
    anchor="w"
)

tk.Label(
    profile_card,
    text="Harshit Upadhyay   •   Data Science Student",
    font=("Segoe UI", 11),
    bg=CARD,
    fg=MUTED
).pack(
    anchor="w",
    pady=(5, 0)
)


# =========================
# PERFORMANCE CARDS
# =========================

performance_frame = tk.Frame(
    content_frame,
    bg=BG
)

performance_frame.pack(
    fill="x",
    padx=25,
    pady=10
)


def create_score_card(
    parent,
    subject,
    score,
    message,
    color
):

    card = tk.Frame(
        parent,
        bg=CARD,
        padx=20,
        pady=15
    )

    card.pack(
        side="left",
        fill="both",
        expand=True,
        padx=5
    )

    tk.Label(
        card,
        text=subject,
        font=("Segoe UI", 12, "bold"),
        bg=CARD,
        fg=TEXT
    ).pack()

    tk.Label(
        card,
        text=str(score),
        font=("Segoe UI", 25, "bold"),
        bg=CARD,
        fg=color
    ).pack(
        pady=2
    )

    tk.Label(
        card,
        text=message,
        font=("Segoe UI", 9),
        bg=CARD,
        fg=color
    ).pack()


create_score_card(
    performance_frame,
    "Machine Learning",
    48,
    "Needs Attention",
    DANGER
)

create_score_card(
    performance_frame,
    "Statistics",
    55,
    "Needs Improvement",
    WARNING
)

create_score_card(
    performance_frame,
    "Power BI",
    80,
    "Good",
    SUCCESS
)


# =========================
# QUESTION CARD
# =========================

question_card = tk.Frame(
    content_frame,
    bg=CARD,
    padx=25,
    pady=20
)

question_card.pack(
    fill="x",
    padx=25,
    pady=15
)

tk.Label(
    question_card,
    text="💬  Ask your Education Agent",
    font=("Segoe UI", 16, "bold"),
    bg=CARD,
    fg=TEXT
).pack(
    anchor="w"
)


question_box = tk.Text(
    question_card,
    height=4,
    font=("Segoe UI", 11),
    bg="#F8FAFC",
    fg=TEXT,
    relief="flat",
    wrap="word",
    padx=10,
    pady=10
)

question_box.pack(
    fill="x",
    pady=10
)

question_box.insert(
    "1.0",
    "What are the student's weak subjects and what should they study?"
)


generate_button = tk.Button(
    question_card,
    text="✨ Generate Study Plan",
    font=("Segoe UI", 12, "bold"),
    bg=PRIMARY,
    fg="white",
    activebackground=PRIMARY,
    activeforeground="white",
    relief="flat",
    cursor="hand2",
    padx=20,
    pady=10,
    command=generate_plan
)

generate_button.pack(
    pady=5
)


status_label = tk.Label(
    question_card,
    text="Ready to analyze your learning performance",
    font=("Segoe UI", 10),
    bg=CARD,
    fg=MUTED
)

status_label.pack(
    pady=(8, 0)
)


# =========================
# RESULT CARD
# =========================

result_card = tk.Frame(
    content_frame,
    bg=CARD,
    padx=25,
    pady=20
)

result_card.pack(
    fill="both",
    expand=True,
    padx=25,
    pady=(5, 25)
)


tk.Label(
    result_card,
    text="🤖  AI Personalized Study Plan",
    font=("Segoe UI", 16, "bold"),
    bg=CARD,
    fg=TEXT
).pack(
    anchor="w",
    pady=(0, 10)
)


# Result area with its own scrollbar
result_frame = tk.Frame(
    result_card,
    bg="#F8FAFC"
)

result_frame.pack(
    fill="both",
    expand=True
)


result_scrollbar = ttk.Scrollbar(
    result_frame,
    orient="vertical"
)

result_scrollbar.pack(
    side="right",
    fill="y"
)


result_box = tk.Text(
    result_frame,
    height=18,
    font=("Segoe UI", 11),
    bg="#F8FAFC",
    fg=TEXT,
    relief="flat",
    wrap="word",
    padx=15,
    pady=15,
    yscrollcommand=result_scrollbar.set
)

result_box.pack(
    side="left",
    fill="both",
    expand=True
)

result_scrollbar.config(
    command=result_box.yview
)


result_box.insert(
    "1.0",
    "Your personalized AI study plan will appear here.\n\n"
    "Click 'Generate Study Plan' to begin."
)

result_box.config(
    state="disabled"
)


# =========================
# FOOTER
# =========================

tk.Label(
    content_frame,
    text="Powered by RAG + Ollama Embeddings + Llama 3.2",
    font=("Segoe UI", 9),
    bg=BG,
    fg=MUTED
).pack(
    pady=(0, 20)
)


# =========================
# START APPLICATION
# =========================

window.mainloop()