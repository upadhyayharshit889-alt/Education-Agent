import tkinter as tk
from tkinter import ttk
import threading
import numpy as np

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================================================
# AI + RAG
# =========================================================

with open("student_data.txt", "r", encoding="utf-8") as f:
    student_data = f.read()

chunks = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
).split_text(student_data)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0

    return np.dot(a, b) / denominator


def generate_ai(question):

    # RAG retrieval
    question_embedding = embeddings.embed_query(question)
    chunk_embeddings = embeddings.embed_documents(chunks)

    scores = [
        cosine_similarity(question_embedding, emb)
        for emb in chunk_embeddings
    ]

    best_chunk = chunks[np.argmax(scores)]

    # Llama
    llm = ChatOllama(
        model="llama3.2",
        temperature=0
    )

    prompt = f"""
You are Education Agent, an AI-powered personalized
learning assistant.

Student Information:
{student_data}

Relevant Retrieved Information:
{best_chunk}

Student Question:
{question}

Analyze the student and provide:

1. Weak Subjects
2. Priority Level
3. Topics to Study
4. Personalized Study Plan
5. Daily Practice

Give a simple, clear and practical answer.
"""

    response = llm.invoke(prompt)

    return response.content


# =========================================================
# COLORS
# =========================================================

BG = "#F5F7FB"
WHITE = "#FFFFFF"
DARK = "#172033"
GRAY = "#667085"
PURPLE = "#4F46E5"
PURPLE_LIGHT = "#EEF2FF"
GREEN = "#16A34A"
GREEN_LIGHT = "#ECFDF3"
ORANGE = "#F59E0B"
ORANGE_LIGHT = "#FFF7E6"
RED = "#DC2626"
RED_LIGHT = "#FEF2F2"
BLUE = "#2563EB"


# =========================================================
# MAIN WINDOW
# =========================================================

root = tk.Tk()

root.title("Education Agent - AI Learning Dashboard")
root.geometry("1200x850")
root.minsize(950, 700)
root.configure(bg=BG)


# =========================================================
# HEADER
# =========================================================

header = tk.Frame(
    root,
    bg=PURPLE,
    height=90
)

header.pack(fill="x")
header.pack_propagate(False)


left_header = tk.Frame(
    header,
    bg=PURPLE
)

left_header.pack(
    side="left",
    padx=30,
    pady=15
)

tk.Label(
    left_header,
    text="🎓  EDUCATION AGENT",
    font=("Segoe UI", 23, "bold"),
    bg=PURPLE,
    fg=WHITE
).pack(anchor="w")

tk.Label(
    left_header,
    text="AI-Powered Personalized Learning Assistant",
    font=("Segoe UI", 10),
    bg=PURPLE,
    fg="#E0E7FF"
).pack(anchor="w")


status_header = tk.Label(
    header,
    text="●  AI SYSTEM READY",
    font=("Segoe UI", 10, "bold"),
    bg=PURPLE,
    fg="#BBF7D0"
)

status_header.pack(
    side="right",
    padx=30
)


# =========================================================
# SCROLLABLE DASHBOARD
# =========================================================

main_canvas = tk.Canvas(
    root,
    bg=BG,
    highlightthickness=0
)

main_scroll = ttk.Scrollbar(
    root,
    orient="vertical",
    command=main_canvas.yview
)

main_canvas.configure(
    yscrollcommand=main_scroll.set
)

main_scroll.pack(
    side="right",
    fill="y"
)

main_canvas.pack(
    side="left",
    fill="both",
    expand=True
)


dashboard = tk.Frame(
    main_canvas,
    bg=BG
)

canvas_window = main_canvas.create_window(
    (0, 0),
    window=dashboard,
    anchor="nw"
)


def update_scroll(event=None):
    main_canvas.configure(
        scrollregion=main_canvas.bbox("all")
    )


dashboard.bind(
    "<Configure>",
    update_scroll
)


def resize_dashboard(event):
    main_canvas.itemconfig(
        canvas_window,
        width=event.width
    )


main_canvas.bind(
    "<Configure>",
    resize_dashboard
)


# =========================================================
# WELCOME SECTION
# =========================================================

welcome = tk.Frame(
    dashboard,
    bg=BG
)

welcome.pack(
    fill="x",
    padx=30,
    pady=(25, 10)
)

tk.Label(
    welcome,
    text="Good Morning, Harshit 👋",
    font=("Segoe UI", 22, "bold"),
    bg=BG,
    fg=DARK
).pack(anchor="w")

tk.Label(
    welcome,
    text="Here is your personalized learning overview.",
    font=("Segoe UI", 11),
    bg=BG,
    fg=GRAY
).pack(anchor="w")


# =========================================================
# PROFILE CARD
# =========================================================

profile = tk.Frame(
    dashboard,
    bg=WHITE,
    padx=25,
    pady=20
)

profile.pack(
    fill="x",
    padx=30,
    pady=10
)

profile_left = tk.Frame(
    profile,
    bg=WHITE
)

profile_left.pack(
    side="left"
)

tk.Label(
    profile_left,
    text="👤",
    font=("Segoe UI", 28),
    bg=WHITE
).pack(side="left", padx=(0, 15))

info = tk.Frame(
    profile_left,
    bg=WHITE
)

info.pack(side="left")

tk.Label(
    info,
    text="Harshit Upadhyay",
    font=("Segoe UI", 16, "bold"),
    bg=WHITE,
    fg=DARK
).pack(anchor="w")

tk.Label(
    info,
    text="Data Science Student  •  Personalized Learning",
    font=("Segoe UI", 10),
    bg=WHITE,
    fg=GRAY
).pack(anchor="w")


tk.Label(
    profile,
    text="AI LEARNING PROFILE",
    font=("Segoe UI", 9, "bold"),
    bg=PURPLE_LIGHT,
    fg=PURPLE,
    padx=12,
    pady=7
).pack(side="right")


# =========================================================
# SECTION TITLE
# =========================================================

tk.Label(
    dashboard,
    text="📊 Performance Overview",
    font=("Segoe UI", 17, "bold"),
    bg=BG,
    fg=DARK
).pack(
    anchor="w",
    padx=30,
    pady=(15, 10)
)


# =========================================================
# PERFORMANCE CARDS
# =========================================================

cards = tk.Frame(
    dashboard,
    bg=BG
)

cards.pack(
    fill="x",
    padx=25
)


def score_card(parent, subject, score, status, color, light):

    card = tk.Frame(
        parent,
        bg=WHITE,
        padx=20,
        pady=18
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
        font=("Segoe UI", 11, "bold"),
        bg=WHITE,
        fg=DARK
    ).pack(anchor="w")

    tk.Label(
        card,
        text=score,
        font=("Segoe UI", 30, "bold"),
        bg=WHITE,
        fg=color
    ).pack(anchor="w", pady=(8, 2))

    # Progress bar
    bar_bg = tk.Frame(
        card,
        bg="#E5E7EB",
        height=7
    )

    bar_bg.pack(
        fill="x",
        pady=(3, 10)
    )

    bar = tk.Frame(
        bar_bg,
        bg=color,
        height=7
    )

    bar.pack(
        side="left",
        fill="y",
        expand=False,
        padx=0
    )

    # Width based on score
    card.update_idletasks()

    bar.config(
        width=max(20, int(card.winfo_width() * int(score) / 100))
    )

    tk.Label(
        card,
        text=status,
        font=("Segoe UI", 9, "bold"),
        bg=light,
        fg=color,
        padx=8,
        pady=4
    ).pack(anchor="w")


score_card(
    cards,
    "Machine Learning",
    48,
    "⚠ Needs Attention",
    RED,
    RED_LIGHT
)

score_card(
    cards,
    "Statistics",
    55,
    "▲ Needs Improvement",
    ORANGE,
    ORANGE_LIGHT
)

score_card(
    cards,
    "Power BI",
    80,
    "✓ Good Performance",
    GREEN,
    GREEN_LIGHT
)


# =========================================================
# AI INSIGHT
# =========================================================

insight = tk.Frame(
    dashboard,
    bg=PURPLE_LIGHT,
    padx=22,
    pady=18
)

insight.pack(
    fill="x",
    padx=30,
    pady=20
)

tk.Label(
    insight,
    text="✨ AI INSIGHT",
    font=("Segoe UI", 11, "bold"),
    bg=PURPLE_LIGHT,
    fg=PURPLE
).pack(anchor="w")

tk.Label(
    insight,
    text="Machine Learning and Statistics need more attention. "
         "Focus your study time on these subjects first.",
    font=("Segoe UI", 11),
    bg=PURPLE_LIGHT,
    fg=DARK,
    wraplength=1000,
    justify="left"
).pack(anchor="w", pady=(5, 0))


# =========================================================
# ASK AI CARD
# =========================================================

ask_card = tk.Frame(
    dashboard,
    bg=WHITE,
    padx=25,
    pady=22
)

ask_card.pack(
    fill="x",
    padx=30,
    pady=5
)

tk.Label(
    ask_card,
    text="🤖 Ask Your Education Agent",
    font=("Segoe UI", 17, "bold"),
    bg=WHITE,
    fg=DARK
).pack(anchor="w")

tk.Label(
    ask_card,
    text="Ask a question about your learning performance.",
    font=("Segoe UI", 10),
    bg=WHITE,
    fg=GRAY
).pack(anchor="w", pady=(3, 12))


question = tk.Text(
    ask_card,
    height=3,
    font=("Segoe UI", 11),
    bg="#F8FAFC",
    fg=DARK,
    relief="flat",
    wrap="word",
    padx=12,
    pady=10
)

question.pack(
    fill="x"
)

question.insert(
    "1.0",
    "What are the student's weak subjects and what should they study?"
)


# =========================================================
# GENERATE BUTTON
# =========================================================

button = tk.Button(
    ask_card,
    text="✨  Generate Personalized Study Plan",
    font=("Segoe UI", 11, "bold"),
    bg=PURPLE,
    fg=WHITE,
    activebackground="#4338CA",
    activeforeground=WHITE,
    relief="flat",
    cursor="hand2",
    padx=22,
    pady=11
)

button.pack(
    pady=(14, 5)
)


# =========================================================
# STATUS
# =========================================================

status = tk.Label(
    ask_card,
    text="● Ready to analyze your learning performance",
    font=("Segoe UI", 9),
    bg=WHITE,
    fg=GRAY
)

status.pack()


# =========================================================
# RESULT SECTION
# =========================================================

tk.Label(
    dashboard,
    text="🧠 AI Personalized Study Plan",
    font=("Segoe UI", 17, "bold"),
    bg=BG,
    fg=DARK
).pack(
    anchor="w",
    padx=30,
    pady=(25, 10)
)


result_card = tk.Frame(
    dashboard,
    bg=WHITE,
    padx=20,
    pady=20
)

result_card.pack(
    fill="both",
    padx=30,
    pady=(0, 25)
)


result = tk.Text(
    result_card,
    height=18,
    font=("Segoe UI", 11),
    bg="#F8FAFC",
    fg=DARK,
    relief="flat",
    wrap="word",
    padx=15,
    pady=15
)

result.pack(
    fill="both",
    expand=True
)

result.insert(
    "1.0",
    """Your AI-generated personalized study plan will appear here.

Click "Generate Personalized Study Plan" to begin.

The Education Agent will:
• Analyze your performance
• Retrieve relevant information using RAG
• Identify weak subjects
• Generate a personalized learning plan"""
)

result.config(
    state="disabled"
)


# =========================================================
# AI GENERATION
# =========================================================

def generate_plan():

    q = question.get(
        "1.0",
        tk.END
    ).strip()

    if not q:
        return

    button.config(
        state="disabled",
        text="⏳  AI is Thinking..."
    )

    status.config(
        text="● Analyzing student performance...",
        fg=PURPLE
    )

    status_header.config(
        text="● AI ANALYZING...",
        fg="#FDE68A"
    )

    result.config(
        state="normal"
    )

    result.delete(
        "1.0",
        tk.END
    )

    result.insert(
        "1.0",
        """🤖 Education Agent is analyzing your learning data...

🔎 Retrieving relevant information...
🧠 Generating personalized recommendations...

Please wait..."""
    )

    result.config(
        state="disabled"
    )

    def worker():

        try:
            answer = generate_ai(q)
        except Exception as e:
            answer = "Error:\n\n" + str(e)

        root.after(
            0,
            lambda: show_result(answer)
        )

    threading.Thread(
        target=worker,
        daemon=True
    ).start()


def show_result(answer):

    result.config(
        state="normal"
    )

    result.delete(
        "1.0",
        tk.END
    )

    result.insert(
        "1.0",
        answer
    )

    result.config(
        state="disabled"
    )

    button.config(
        state="normal",
        text="✨  Generate Personalized Study Plan"
    )

    status.config(
        text="● Personalized study plan generated successfully",
        fg=GREEN
    )

    status_header.config(
        text="● AI SYSTEM READY",
        fg="#BBF7D0"
    )


button.config(
    command=generate_plan
)


# =========================================================
# FOOTER
# =========================================================

tk.Label(
    dashboard,
    text="Education Agent  •  RAG + Ollama + Llama 3.2",
    font=("Segoe UI", 9),
    bg=BG,
    fg=GRAY
).pack(
    pady=(0, 25)
)


# =========================================================
# START
# =========================================================

root.mainloop()