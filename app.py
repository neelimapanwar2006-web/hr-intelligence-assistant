"""
HR Intelligence Assistant — Gradio Chat Interface
==================================================
Launches a browser-based chat UI for the HR RAG assistant.
Author: Neelima Panwar

Run:  python src/app.py
"""

import os
import gradio as gr
from dotenv import load_dotenv
from hr_assistant import load_documents, build_vectorstore, build_qa_chain, ask

load_dotenv()

DEFAULT_HR_PDF = os.getenv(
    "HR_POLICY_PDF",
    "https://www.nestle.com/sites/default/files/asset-library/documents/jobs/the_nestle_hr_policy_pdf_2012.pdf"
)

# ── Initialise once at startup ────────────────────────────────────────────────
print("Initialising HR Assistant...")
documents   = load_documents(DEFAULT_HR_PDF)
vectorstore = build_vectorstore(documents)
qa_chain    = build_qa_chain(vectorstore)
print("Ready.")


# ── Gradio chat function ──────────────────────────────────────────────────────
def respond(message: str, history: list) -> str:
    """Process user message and return assistant response."""
    if not message.strip():
        return "Please enter a question."
    try:
        result = ask(qa_chain, message)
        answer = result["answer"]

        # Append source references if available
        sources = list(set(result.get("sources", [])))
        if sources:
            answer += "\n\n📄 **Source:** " + ", ".join(sources)

        return answer
    except Exception as e:
        print(f"Error: {e}")
        return "Sorry, something went wrong. Please try again."


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="HR Intelligence Assistant") as demo:

    gr.Markdown("""
    # 🤖 HR Intelligence Assistant
    Ask questions about HR policies, leave rules, grievance procedures, and more.
    Answers are grounded in the official HR policy document.
    """)

    chatbot = gr.ChatInterface(
        fn=respond,
        examples=[
            "What is the maternity leave policy?",
            "How do I raise a grievance?",
            "What are the best practices for conducting interviews?",
            "What is the code of conduct for employees?",
        ],
        retry_btn=None,
        undo_btn=None,
    )

demo.launch(debug=False, share=False)
