from langchain_ollama import ChatOllama

# Student's weak areas detected by our program
weak_areas = """
Machine Learning - 48 marks - HIGH priority
Statistics - 55 marks - MEDIUM priority
Data Visualization - Missing from study material - MEDIUM priority
"""

# Connect to our local Ollama AI
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

# Ask AI to create the study plan
prompt = f"""
You are an education study-planning assistant.

Create a simple 7-day study plan for this student.

Weak areas:
{weak_areas}

Rules:
- Give more time to HIGH priority topics.
- Keep the plan beginner-friendly.
- Include what to study and what to practice.
- Do not invent marks or subjects.
- Use only the weak areas provided above.

Give the answer in a clear day-by-day format.
"""

response = llm.invoke(prompt)

print("===== AI GENERATED STUDY PLAN =====")
print(response.content)