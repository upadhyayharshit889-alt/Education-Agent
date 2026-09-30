import tkinter as tk
from tkinter import scrolledtext
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
import math


# ---------- RAG ----------
with open("student_data.txt", "r", encoding="utf-8") as file:
    text = file.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text"
)

chunk_embeddings = [
    embeddings_model.embed_query(chunk)
    for chunk in chunks
]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0

    return dot / (norm_a * norm_b)


# ---------- AI ----------
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


def generate_plan():
    question = question_box.get("1.0", tk.END).strip()

    if not question:
        question = "What are the student's weak subjects?"

    question_embedding = embeddings_model.embed_query(question)

    scores = []

    for i, embedding in enumerate(chunk_embeddings):
        score = cosine_similarity(
            question_embedding,
            embedding
        )
        scores.append((score, i))

    scores.sort(reverse=True)

    retrieved_content = chunks[scores[0][1]]

    prompt = f"""
You are an AI Education Agent.

Student information:
{retrieved_content}

User question:
{question}

Give a personalized answer.

Include:
1. Weak subjects
2. Topics to study
3. Priority
4. Simple study plan

Keep the answer clear and practical.
"""

    response = llm.invoke(prompt)

    output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, response.content)


# ---------- UI ----------
window = tk.Tk()

window.title("🎓 Education Agent")
window.geometry("900x650")

title = tk.Label(
    window,
    text="🎓 EDUCATION AGENT",
    font=("Arial", 24, "bold")
)

title.pack(pady=15)

subtitle = tk.Label(
    window,
    text="RAG + AI Personalized Study Planner",
    font=("Arial", 13)
)

subtitle.pack()

question_box = scrolledtext.ScrolledText(
    window,
    height=4,
    font=("Arial", 12)
)

question_box.pack(
    padx=30,
    pady=20,
    fill="x"
)

question_box.insert(
    tk.END,
    "What are the student's weak subjects and what should they study?"
)

button = tk.Button(
    window,
    text="🤖 Generate Study Plan",
    font=("Arial", 14, "bold"),
    command=generate_plan
)

button.pack(pady=10)

output_box = scrolledtext.ScrolledText(
    window,
    height=20,
    font=("Arial", 12)
)

output_box.pack(
    padx=30,
    pady=15,
    fill="both",
    expand=True
)

window.mainloop()