from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import math

# 1. Load student material
with open("student_data.txt", "r", encoding="utf-8") as file:
    text = file.read()

print("Student material loaded")

# 2. Split material into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print("Number of chunks:", len(chunks))

# 3. Create embedding model
embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text"
)

# 4. Create embeddings for each chunk
print("Creating embeddings...")

chunk_embeddings = []

for i, chunk in enumerate(chunks):
    print(f"Embedding chunk {i + 1}/{len(chunks)}...")
    embedding = embeddings_model.embed_query(chunk)
    chunk_embeddings.append(embedding)

print("All embeddings created")


# 5. Ask a question
question = "What are the student's weak subjects?"

print("\nQuestion:", question)

question_embedding = embeddings_model.embed_query(question)


# 6. Cosine similarity function
def cosine_similarity(a, b):

    dot = sum(x * y for x, y in zip(a, b))

    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


# 7. Compare question with all chunks
scores = []

for i, embedding in enumerate(chunk_embeddings):

    score = cosine_similarity(
        question_embedding,
        embedding
    )

    scores.append((score, i))


# 8. Sort by similarity
scores.sort(reverse=True)


# 9. Display best results
print("\n===== RAG SEARCH RESULT =====")

top_k = min(3, len(scores))

for rank in range(top_k):

    score, index = scores[rank]

    print(f"\n--- Result {rank + 1} ---")
    print("Similarity:", round(score, 4))
    print("Content:")
    print(chunks[index])


print("\n===== RAG TEST COMPLETED =====")

from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

retrieved_content = chunks[scores[0][1]]

prompt = f"""
You are an AI education assistant.

Student information:
{retrieved_content}

Tell me:
1. What are the weak subjects?
2. What should the student study?
3. Create a simple study plan.

Keep the answer simple.
"""

print("\n===== AI STUDY PLAN =====")

response = llm.invoke(prompt)

print(response.content)

# 10. Send retrieved information to AI

from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

retrieved_content = chunks[scores[0][1]]

prompt = f"""
You are an AI education assistant.

Student information:
{retrieved_content}

Based on the student information:

1. Identify the weak subjects.
2. Identify important topics to study.
3. Create a simple study plan.
4. Give priority to the weakest subjects.

Keep the answer simple and practical.
"""

print("\n===== AI STUDY PLAN =====")

response = llm.invoke(prompt)

print(response.content)

print("\n===== EDUCATION AGENT COMPLETED =====")