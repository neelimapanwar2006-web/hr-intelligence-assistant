# HR Intelligence Assistant 🤖
### RAG-Based HR Policy Q&A System using LangChain + FAISS + OpenAI

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5-412991?logo=openai&logoColor=white)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-orange)
![Gradio](https://img.shields.io/badge/UI-Gradio-ff7c00)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

The **HR Intelligence Assistant** lets employees ask natural language questions about HR policies and get precise, grounded answers — powered by a **Retrieval-Augmented Generation (RAG)** pipeline.

Instead of searching through hundreds of pages of policy documents, employees simply type a question and get an accurate, context-aware response with source references in seconds.

> Built as a real-world agentic AI project by **Neelima Panwar**, Data Scientist & AI Developer.

---

## Demo

```
You: What is the maternity leave policy?
Assistant: According to the HR policy document, employees are entitled to
maternity leave as outlined in Section 4.2... 

📄 Source: the_nestle_hr_policy_pdf_2012.pdf
```

---

## Key Features

- **Semantic Search** — Retrieves the most relevant policy sections using vector embeddings, not simple keyword matching
- **Grounded Answers** — Responses are generated only from retrieved context, reducing hallucination
- **Source Citations** — Every answer references the document it was drawn from
- **Conversational UI** — Built with Gradio for a clean, browser-based chat experience
- **Configurable** — Swap in any HR policy PDF via a simple environment variable
- **CLI + Web Modes** — Run as a terminal Q&A tool or launch the full chat interface

---

## Architecture

```
User Question
     │
     ▼
┌─────────────────────────┐
│   RetrievalQA Chain      │
│   (LangChain)            │
└──────────┬───────────────┘
           │
           ▼
┌─────────────────────────┐       ┌──────────────────────┐
│   FAISS Vector Store     │◄──────│  HR Policy PDF        │
│   (semantic retrieval)   │       │  (chunked + embedded) │
└──────────┬───────────────┘       └──────────────────────┘
           │
           ▼
┌─────────────────────────┐
│   ChatOpenAI (GPT-3.5)   │
│   + Custom HR Prompt     │
└──────────┬───────────────┘
           │
           ▼
   Grounded Answer + Source
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT-3.5-turbo |
| Orchestration | LangChain (RetrievalQA chain) |
| Embeddings | OpenAI Embeddings |
| Vector Store | FAISS |
| Document Loading | PyPDFLoader |
| Chat UI | Gradio |
| Language | Python 3.10+ |

---

## Project Structure

```
hr-intelligence-assistant/
├── src/
│   ├── hr_assistant.py     # Core RAG pipeline (load → embed → retrieve → answer)
│   └── app.py               # Gradio chat interface
├── data/
│   └── sample_policies/     # Place sample HR policy PDFs here (optional)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- An OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/neelima-panwar/hr-intelligence-assistant.git
cd hr-intelligence-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Run — Command Line Mode

```bash
cd src
python hr_assistant.py
```

Ask questions directly in the terminal. Type `exit` to quit.

### Run — Web Chat Interface

```bash
cd src
python app.py
```

This launches a Gradio chat UI in your browser (default: `http://localhost:7860`).

### Use Your Own HR Policy Document

By default, the project loads a sample public HR policy PDF. To use your own:

```bash
# In .env, set:
HR_POLICY_PDF=https://your-url-or-local-path/your-policy.pdf
```

---

## Example Queries

- "What is the maternity leave policy?"
- "How do I raise a grievance?"
- "What are the best practices for conducting interviews?"
- "What is the code of conduct for employees?"

---

## Future Improvements

- [ ] Multi-document support (ingest multiple policy PDFs at once)
- [ ] Add SQL-based employee data agent (tool-calling) for queries like "how many leaves does X have left"
- [ ] Conversational memory across multi-turn sessions
- [ ] Migrate from LangChain to LangGraph for more robust agent orchestration
- [ ] Deploy as a REST API (FastAPI) for integration into internal HR portals
- [ ] Add CI/CD pipeline for automated testing and deployment

---

## About the Author

**Neelima Panwar** — Data Scientist & AI Developer
6+ years of experience in data analytics, machine learning, and agentic AI systems.
Published researcher in ML (Scopus, Springer).

[LinkedIn](https://linkedin.com/in/neelima-panwar) · [Email](mailto:neelimapanwar2006@gmail.com)

---

## License

MIT License — feel free to use and adapt with attribution.
