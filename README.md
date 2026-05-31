# DocuSense 🌌

DocuSense is a smart AI assistant web application that helps you interact with your documents and the live internet at the same time. 

Instead of reading through a massive PDF file to find a single piece of information, you can simply upload your document to DocuSense and chat with it like a human. Additionally, if you need fresh information that isn't inside the document, DocuSense can autonomously browse the live internet to create a detailed report for you.

---

## 🚀 Key Features

The application is split into two main sections (tabs):

1. **Tab 1: Chat with PDF (RAG Engine)**
   * Upload any PDF document (agreements, notes, company policies).
   * Ask questions in plain language.
   * Receive answers based **only** on the uploaded document (the AI will not make things up or hallucinate).
   * View clear citations showing exactly which page the information came from.

2. **Tab 2: Corporate Intelligence (Web Agent)**
   * Type in the name of any company or topic.
   * The AI deploys an automated web scout to search the live internet.
   * It gathers the latest news and facts, formatting them into a clean, easy-to-read Executive Briefing.

---

## 🧠 How It Works (Under the Hood)

Here is a simple breakdown of how the data flows through the application:

### The PDF Chat Process
* **Reading & Slicing:** When you upload a PDF, the app reads the raw text and cuts it into smaller chunks (about 1,000 characters each) so the AI can process it easily.
* **Turning Text into Math (Embeddings):** A local embedding model translates these text chunks into long lists of numbers (called vectors). This helps the computer understand the actual *meaning* of the words.
* **Smart Search:** When you ask a question, the app converts your question into numbers, searches the vector database, and pulls out the top 3 most relevant paragraphs from your PDF.
* **Generating the Answer:** The app sends those 3 paragraphs along with your question to a lightning-fast AI model (Llama 3.1 running on Groq cloud), which reads the text and writes back a perfect response.

### The Web Search Process
* The app takes your query, searches the live web using DuckDuckGo, extracts raw headlines and descriptions, and uses the AI model to summarize the chaos into a structured corporate profile.

---

## 🛠️ The Tech Stack

This project uses a modern and efficient combination of tools:

* **Streamlit:** Used to build the entire web interface and user interface buttons entirely in Python.
* **LangChain:** The central backbone framework that connects the document readers, prompt templates, and AI models together.
* **Groq API (Llama 3.1 8B):** A cloud-based AI model provider that runs on specialized high-speed chips, making the AI respond almost instantly.
* **HuggingFace (`all-MiniLM-L6-v2`):** A lightweight, open-source model that runs completely locally on the server to turn text into mathematical vectors for free.
* **DuckDuckGo Search Tool:** An open-source web scraper that allows the AI to browse live internet results without needing complex API setup.

---

## ⚙️ How to Setup and Run Locally

Follow these straightforward steps to get the project running on your own computer:

### 1. Clone the project
Open your terminal or command prompt and run:
```bash
git clone [https://github.com/alhajbaig/DocuSense-RAG-implemented-.git](https://github.com/alhajbaig/DocuSense-RAG-implemented-.git)
cd DocuSense-RAG-implemented-
