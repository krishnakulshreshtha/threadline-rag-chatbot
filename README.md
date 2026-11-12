# Threadline AI – RAG-Powered Customer Support Chatbot

An intelligent AI customer support assistant built for **Threadline Apparel**, a fictional streetwear and clothing brand. Powered by Retrieval-Augmented Generation (RAG), the chatbot answers customer queries using the brand's actual knowledge base rather than relying solely on an LLM's general knowledge.

The system combines semantic retrieval, query expansion, cross-encoder reranking, and conversational context handling to deliver accurate, relevant, and source-grounded responses.

## Overview

Threadline AI acts as a virtual customer support representative, helping customers find information about:

- Products and product specifications
- Sizes, fitting, and available colors
- Product availability and inventory
- Shipping and delivery policies
- Returns and exchanges
- Payments and promotions
- Order support and general store policies

The chatbot retrieves relevant information from the knowledge base before generating responses, reducing unsupported answers and improving factual accuracy.

## Key Features

### 1. Retrieval-Augmented Generation (RAG)
- Loads business knowledge from PDF documents.
- Splits documents into overlapping text chunks.
- Generates semantic embeddings using HuggingFace.
- Stores and retrieves document vectors using ChromaDB.
- Grounds LLM responses in retrieved business information.

### 2. Intelligent Retrieval Pipeline
- **Query Expansion:** Generates alternative phrasings of customer questions to improve retrieval coverage.
- **Semantic Search:** Retrieves relevant document chunks using vector similarity.
- **Cross-Encoder Reranking:** Reorders retrieved candidates using a relevance scoring model.
- **Context Selection:** Passes the highest-ranked documents to the LLM for response generation.

### 3. Conversational Memory
- Supports multi-turn conversations.
- Rewrites follow-up questions into standalone queries using previous conversation context.
- Maintains conversation history for context-aware responses.

### 4. Source Attribution
- Identifies the documents used to generate an answer.
- Displays source filenames and relevant text previews.
- Avoids fabricating sources when no supporting documents are retrieved.

### 5. FastAPI Backend
- Exposes a REST API for chatbot interaction.
- Accepts customer messages and conversation history.
- Returns AI-generated responses along with retrieved source information.
- Uses asynchronous request handling with a thread pool for synchronous RAG operations.

## Architecture

```text
              Customer
                  |
                  v
          Customer Query
                  |
                  v
          FastAPI Backend
                  |
                  v
       Conversation Context
                  |
                  v
          Query Rewriting
                  |
                  v
           Query Expansion
                  |
                  v
          ChromaDB Retrieval
                  |
                  v
        Cross-Encoder Reranking
                  |
                  v
          Top Relevant Chunks
                  |
                  v
         Context + Conversation
                  |
                  v
           GPT-4.1-mini
                  |
                  v
        Grounded AI Response
                  |
          +-------+-------+
          |               |
          v               v
     Chat Response    Source Citations
```

## Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Backend | FastAPI |
| LLM | OpenAI GPT-4.1-mini |
| LLM Framework | LangChain |
| Embedding Model | HuggingFace all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| Reranking | Sentence Transformers Cross-Encoder |
| Document Processing | PyPDF |
| Text Splitting | RecursiveCharacterTextSplitter |
| API Server | Uvicorn |

## Retrieval Configuration

| Parameter | Configuration |
|---|---|
| Chunk Size | 1000 characters |
| Chunk Overlap | 200 characters |
| Initial Retrieval | 40 documents per query |
| Query Expansion | Original query + 2 alternatives |
| Final Context | Top 4 reranked documents |
| Embedding Model | all-MiniLM-L6-v2 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | gpt-4.1-mini |

## Project Structure

```text
threadline-rag-chatbot/
│
├── api.py                  # FastAPI application and chat endpoint
├── csr.py                  # RAG pipeline and chatbot logic
├── requirements.txt        # Python dependencies
│
├── kb2/                    # PDF knowledge base
│
├── vector_db1/             # Persistent ChromaDB storage
│
└── .env                    # API credentials (create locally)
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/krishnakulshreshtha/threadline-rag-chatbot.git

cd threadline-rag-chatbot
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**macOS / Linux**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
```

Replace the placeholder with your own OpenAI API key.

Never commit your `.env` file or expose API credentials publicly.

### 5. Run the FastAPI Server

```bash
uvicorn api:app --reload
```

The backend will be available at:

```text
http://127.0.0.1:8000
```

### 6. Access API Documentation

FastAPI automatically provides interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## API Usage

### POST `/chat`

Send a customer message along with optional conversation history.

**Request Body**

```json
{
  "message": "What is your return policy?",
  "history": []
}
```

**Request with Conversation History**

```json
{
  "message": "What about exchanges?",
  "history": [
    {
      "role": "user",
      "content": "What is your return policy?"
    },
    {
      "role": "assistant",
      "content": "You can refer to our return policy..."
    }
  ]
}
```

**Response Format**

```json
{
  "response": "Here is the information about our exchange policy...",
  "sources": "**Sources**\n\n1. policy.pdf..."
}
```

The response contains:
- `response`: The generated customer support answer.
- `sources`: Source documents and relevant retrieved content.

## Design Principles

- **Grounded Responses:** Business-specific answers should be supported by retrieved knowledge.
- **Retrieval Quality:** Query expansion and reranking improve the relevance of retrieved context.
- **Context Awareness:** Previous messages help interpret follow-up questions.
- **Transparency:** Source information helps identify where answers originated.
- **Reliability:** When the knowledge base lacks sufficient information, the assistant is instructed not to guess.

## Future Improvements

- Real-time product inventory and order status integrations.
- CRM integration for customer and lead management.
- Automated appointment scheduling.
- Website-embeddable chatbot widget.
- LLM provider fallback and error handling.
- Cloud deployment using AWS or Microsoft Azure.
- Retrieval evaluation using MRR, nDCG, and other ranking metrics.

## Author

**Krishna Kulshreshtha**

B.Tech – Computer Science and Engineering

- GitHub: https://github.com/krishnakulshreshtha
- LinkedIn: https://www.linkedin.com/in/krishna-kulshreshtha-bb2732173/

---

*Built as a practical application of Retrieval-Augmented Generation, semantic search, LLM integration, and AI-powered customer support.*
