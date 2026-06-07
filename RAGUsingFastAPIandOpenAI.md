### **Introduction to RAG Using FastAPI and OpenAI**

### **1. What is RAG?**

**RAG** stands for **Retrieval-Augmented Generation**.

In simple terms, **RAG** is a technique where an AI application first **searches for relevant information** from your own documents or database, then uses an AI model to **generate an answer based on that information**.

Instead of asking the AI model to answer from memory only, we give it extra knowledge before it responds.

### **Simple Analogy**

Imagine a student is asked a question in class.

**Without RAG:**

> The student answers using only what they remember.

**With RAG:**

> The student first checks the textbook, finds the correct section, reads it, and then answers the question.

That is what RAG does.

It helps AI applications answer from your own content such as:

- PDF documents
- Company policies
- Training notes
- School notes
- Legal documents
- Product manuals
- Website content
- Database records

---

### **2. Why Do We Need RAG?**

Normal AI models are powerful, but they have limitations.

They may:

- Not know your private documents
- Give outdated information
- Hallucinate, meaning they may confidently give wrong answers
- Fail to cite where the answer came from
- Struggle with company-specific or school-specific content

RAG solves this by connecting the model to a knowledge source.

For example, instead of asking:

> “What is LuxDevHQ’s refund policy?”

and expecting the model to guess, RAG allows the system to search your real policy document first, then answer based on that document.

---

### **3. Main Components of a RAG System**

A simple RAG application has five main parts:

| Component | Meaning |
|---|---|
| User question | The question asked by the user |
| Documents | The knowledge source, such as PDFs or text files |
| Embeddings | Numerical representations of text used for similarity search |
| Vector database | Stores embeddings and helps find the most relevant text |
| LLM | The AI model that generates the final answer |

OpenAI provides embedding models for converting text into vectors, which are useful for search and retrieval tasks.

FastAPI is a Python web framework used to build APIs quickly using Python type hints.

---

### **4. What is FastAPI?**

**FastAPI** is a Python framework used to build APIs.

An API allows other applications to communicate with your backend.

For example, you can build an endpoint like:

```text
POST /ask
```

A user sends a question to this endpoint, and your FastAPI application returns an AI-generated answer.

FastAPI is useful for RAG because it allows us to expose our RAG system as a backend service.

The official FastAPI tutorial shows a basic app created using `FastAPI()` and route decorators such as `@app.get("/")`.

---

### **5. What is OpenAI Used for in RAG?**

In this beginner project, OpenAI will be used for two things:

| Use | Description |
|---|---|
| Embeddings | Convert text into numerical vectors |
| Answer generation | Generate the final response using the retrieved context |

The general flow is:

```text
User asks question
        ↓
Convert question into embedding
        ↓
Search similar document chunks
        ↓
Send question + relevant chunks to OpenAI
        ↓
Return final answer
```

---

### **6. RAG Architecture**

```text
                ┌──────────────────────┐
                │      User Question    │
                └──────────┬───────────┘
                           │
                           ↓
                ┌──────────────────────┐
                │      FastAPI App      │
                └──────────┬───────────┘
                           │
                           ↓
                ┌──────────────────────┐
                │ Create Query Embedding│
                └──────────┬───────────┘
                           │
                           ↓
                ┌──────────────────────┐
                │ Search Vector Store   │
                └──────────┬───────────┘
                           │
                           ↓
                ┌──────────────────────┐
                │ Retrieve Context      │
                └──────────┬───────────┘
                           │
                           ↓
                ┌──────────────────────┐
                │ OpenAI Generates Ans. │
                └──────────┬───────────┘
                           │
                           ↓
                ┌──────────────────────┐
                │ Return Answer to User │
                └──────────────────────┘
```

---

### **7. Beginner Project Goal**

We will build a simple API where:

1. We store a few notes inside the Python file.
2. We convert those notes into embeddings.
3. We allow a user to ask a question.
4. The system finds the most relevant note.
5. OpenAI generates an answer using only that note.

This is a beginner-friendly version before adding PDFs, databases, or ChromaDB.

---

### **8. Install Requirements**

Create a project folder:

```bash
mkdir rag_fastapi_openai
cd rag_fastapi_openai
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it.

On macOS/Linux:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install packages:

```bash
pip install fastapi uvicorn openai python-dotenv numpy
```

Create a `.env` file:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

---

### **9. Single-File Beginner RAG App**

Create a file called:

```bash
main.py
```

Add this code:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import os

# Load environment variables from .env file
load_dotenv()

# Create OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create FastAPI app
app = FastAPI(title="Beginner RAG API with FastAPI and OpenAI")


# Sample knowledge base
documents = [
    {
        "title": "What is RAG?",
        "content": "RAG stands for Retrieval-Augmented Generation. It is a technique where an AI system retrieves relevant information from documents before generating an answer."
    },
    {
        "title": "Why use RAG?",
        "content": "RAG helps reduce hallucinations by grounding AI answers in trusted documents. It is useful for company policies, manuals, school notes, and private knowledge bases."
    },
    {
        "title": "What is FastAPI?",
        "content": "FastAPI is a modern Python framework used to build APIs quickly. It is commonly used to create backend services for machine learning and AI applications."
    },
    {
        "title": "What are embeddings?",
        "content": "Embeddings are numerical representations of text. They help computers compare the meaning of sentences, paragraphs, or documents."
    },
    {
        "title": "What is a vector database?",
        "content": "A vector database stores embeddings and allows similarity search. It helps find documents that are semantically close to a user's question."
    }
]


# Request body model
class QuestionRequest(BaseModel):
    question: str


def get_embedding(text: str):
    """
    Converts text into an embedding vector using OpenAI.
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


def cosine_similarity(vec1, vec2):
    """
    Measures how similar two vectors are.
    The higher the score, the more similar they are.
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# Create document embeddings when the app starts
document_embeddings = []

for doc in documents:
    embedding = get_embedding(doc["content"])
    document_embeddings.append({
        "title": doc["title"],
        "content": doc["content"],
        "embedding": embedding
    })


@app.get("/")
def home():
    return {
        "message": "Welcome to the Beginner RAG API using FastAPI and OpenAI"
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question

    # Step 1: Convert user question into an embedding
    question_embedding = get_embedding(question)

    # Step 2: Compare question with all documents
    similarities = []

    for doc in document_embeddings:
        score = cosine_similarity(question_embedding, doc["embedding"])
        similarities.append({
            "title": doc["title"],
            "content": doc["content"],
            "score": score
        })

    # Step 3: Pick the most relevant document
    best_match = max(similarities, key=lambda x: x["score"])

    context = best_match["content"]

    # Step 4: Ask OpenAI to answer using the retrieved context
    prompt = f"""
You are a helpful AI assistant.

Use the context below to answer the user's question.
If the answer is not in the context, say: "I do not have enough information from the provided context."

Context:
{context}

Question:
{question}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return {
        "question": question,
        "matched_document": best_match["title"],
        "similarity_score": float(best_match["score"]),
        "answer": response.output_text
    }
```

---

### **10. Run the FastAPI App**

Run this command:

```bash
uvicorn main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

For the API documentation, open:

```text
http://127.0.0.1:8000/docs
```

FastAPI automatically provides interactive API documentation where students can test endpoints directly in the browser.

---

### **11. Test the `/ask` Endpoint**

Go to:

```text
http://127.0.0.1:8000/docs
```

Open the `/ask` endpoint and test with:

```json
{
  "question": "Why is RAG useful?"
}
```

Expected response:

```json
{
  "question": "Why is RAG useful?",
  "matched_document": "Why use RAG?",
  "similarity_score": 0.82,
  "answer": "RAG is useful because it helps reduce hallucinations by grounding AI answers in trusted documents..."
}
```

The exact score and answer may vary.

---

### **12. Explaining the Code to Beginners**

### **Part 1: Importing Libraries**

```python
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import os
```

These libraries help us:

- Create the API
- Receive user questions
- Call OpenAI
- Load the API key
- Calculate similarity between vectors

### **Part 2: Creating the FastAPI App**

```python
app = FastAPI(title="Beginner RAG API with FastAPI and OpenAI")
```

This creates our web application.

### **Part 3: Creating a Knowledge Base**

```python
documents = [...]
```

This is our small knowledge base.

In a real project, this can be replaced with:

- PDF files
- Word documents
- Database records
- Website content
- Company policies

### **Part 4: Creating Embeddings**

```python
def get_embedding(text: str):
```

This function converts text into numbers.

Those numbers represent the meaning of the text.

For example:

```text
"What is RAG?"
```

and

```text
"Explain Retrieval-Augmented Generation"
```

may have similar embeddings because they have similar meaning.

### **Part 5: Measuring Similarity**

```python
def cosine_similarity(vec1, vec2):
```

This function compares two embeddings.

A higher score means the texts are more similar.

### **Part 6: Retrieving the Best Document**

```python
best_match = max(similarities, key=lambda x: x["score"])
```

This selects the document that is closest to the user’s question.

### **Part 7: Generating the Answer**

```python
response = client.responses.create(
    model="gpt-4.1-mini",
    input=prompt
)
```

This sends the retrieved context and question to OpenAI, then returns the answer.

The OpenAI Responses API is used to generate model responses.

---

### **13. Example Classroom Explanation**

You can explain RAG to students like this:

> RAG is like giving ChatGPT a textbook before asking it an exam question. The model does not just guess. It first reads the relevant part of the textbook, then answers based on that information.

Another analogy:

> A normal chatbot is like a person answering from memory. A RAG chatbot is like a librarian who first searches the library, finds the correct book, reads the correct page, and then answers.

---

### **14. Real-World Examples of RAG**

| Industry | RAG Example |
|---|---|
| Education | Student asks questions from course notes |
| Law | Lawyer searches legal documents |
| Banking | Staff ask questions from compliance policies |
| Healthcare | Doctors search clinical guidelines |
| Customer support | Chatbot answers from product manuals |
| HR | Employees ask questions from company policies |
| Data engineering | Team searches documentation and pipeline notes |

---

### **15. Common Mistakes Beginners Should Avoid**

### **Mistake 1: Sending the Whole Document to the Model**

Bad approach:

```text
Send entire 300-page PDF to OpenAI
```

Better approach:

```text
Split the PDF into chunks, retrieve only the relevant chunks, then send those chunks to OpenAI.
```

### **Mistake 2: Not Checking Retrieved Context**

Sometimes the system may retrieve the wrong chunk. Always inspect the matched document or source.

### **Mistake 3: Allowing the Model to Guess**

Always instruct the model:

```text
If the answer is not in the context, say you do not have enough information.
```

### **Mistake 4: Hardcoding API Keys**

Do not write this directly in code:

```python
api_key = "sk-..."
```

Use a `.env` file instead.

### **Mistake 5: No Source Tracking**

In production, you should return the source document name, page number, or section title so the user can verify the answer.

---

### **16. Next Level After This Beginner Version**

After students understand this simple version, you can upgrade the project by adding:

1. PDF upload
2. Document chunking
3. ChromaDB or FAISS
4. PostgreSQL with pgvector
5. Source citations
6. Streamlit frontend
7. Authentication
8. Deployment on Render, Railway, or a VPS

---

### **17. Simple Assignment for Students**

### **Assignment: Build a Mini RAG API**

Create a FastAPI application that answers questions from your own notes.

### **Requirements**

1. Create at least 10 documents in a Python list.
2. Generate embeddings for each document.
3. Create a `/ask` endpoint.
4. Retrieve the most relevant document.
5. Use OpenAI to generate an answer.
6. Return:
   - User question
   - Matched document title
   - Answer
   - Similarity score

### **Bonus Task**

Add a `/documents` endpoint that returns all available document titles.

Example:

```python
@app.get("/documents")
def list_documents():
    return {
        "documents": [doc["title"] for doc in documents]
    }
```

---

### **18. Summary**

RAG is one of the most important techniques in modern AI application development.

It allows us to build AI systems that:

- Answer from trusted documents
- Reduce hallucinations
- Work with private knowledge
- Provide more accurate responses
- Support real business use cases

FastAPI helps us expose the RAG system as an API, while OpenAI provides the models for embeddings and answer generation.
