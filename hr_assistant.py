"""
HR Intelligence Assistant
=========================
RAG-based HR policy Q&A system using LangChain + FAISS + OpenAI
Author: Neelima Panwar
"""

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME     = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
TEMPERATURE    = float(os.getenv("TEMPERATURE", "0.7"))
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", "100"))

# Default HR policy document (can be overridden via env or argument)
DEFAULT_HR_PDF = os.getenv(
    "HR_POLICY_PDF",
    "https://www.nestle.com/sites/default/files/asset-library/documents/jobs/the_nestle_hr_policy_pdf_2012.pdf"
)

HR_PROMPT_TEMPLATE = """
You are a knowledgeable and professional HR assistant.
Use ONLY the context provided below to answer the question.
If the answer is not found in the context, say:
"I'm sorry, I couldn't find that information in the HR policy documents."

Context:
{context}

Question: {question}

Answer:
"""


# ── Core Functions ────────────────────────────────────────────────────────────

def load_documents(pdf_path: str) -> list:
    """Load HR policy PDF from a URL or local file path."""
    print(f"Loading document: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")
    return documents


def build_vectorstore(documents: list) -> FAISS:
    """Chunk documents and build a FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    texts = splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(texts, embeddings)
    print("Vector store built successfully.")
    return vectorstore


def build_qa_chain(vectorstore: FAISS) -> RetrievalQA:
    """Build the RetrievalQA chain with a custom HR prompt."""
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=HR_PROMPT_TEMPLATE,
    )

    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        temperature=TEMPERATURE,
        openai_api_key=OPENAI_API_KEY
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa_chain


def ask(qa_chain: RetrievalQA, question: str) -> dict:
    """Run a question through the QA chain and return answer + sources."""
    result = qa_chain({"query": question})
    return {
        "answer": result["result"],
        "sources": [doc.metadata.get("source", "Unknown") for doc in result.get("source_documents", [])]
    }


# ── Main (CLI mode) ───────────────────────────────────────────────────────────

def main():
    print("\n🤖 HR Intelligence Assistant — Initialising...\n")

    documents   = load_documents(DEFAULT_HR_PDF)
    vectorstore = build_vectorstore(documents)
    qa_chain    = build_qa_chain(vectorstore)

    print("\n✅ Ready! Type your HR question below. (Type 'exit' to quit)\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        if not question:
            continue

        result = ask(qa_chain, question)
        print(f"\nAssistant: {result['answer']}\n")


if __name__ == "__main__":
    main()
