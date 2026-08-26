
import os
import glob
import time
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.manifold import TSNE
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
import gradio as gr
from sentence_transformers import CrossEncoder
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter


reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


MODEL = "gpt-4.1-mini"
db_name = "vector_db1"
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")



folders = glob.glob("kb2/*.pdf")

documents = []

for folder in folders:

    loader = PyPDFLoader(folder)
    folder_docs = loader.load()

    for doc in folder_docs:
        doc.metadata["doc_type"] = "clothing_brand"
        documents.append(doc)

print(f"Loaded {len(documents)} documents")



text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

if Path(db_name).exists():
    vectorstore = Chroma(
        persist_directory=db_name,
        embedding_function=embeddings
    )
else:
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=db_name
    )
llm = ChatOpenAI(temperature=0, model_name=MODEL)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 40}
)




SYSTEM_PROMPT = """
You are Threadline AI, the customer support assistant for Threadline Apparel, a fictional direct-to-consumer clothing brand specializing in streetwear and casual apparel.

Your job is to help customers with:
- Products
- Product specifications
- Sizes and fit
- Availability and inventory
- Shipping
- Returns and exchanges
- Payments
- Promotions
- Order support
- General Threadline policies

The retrieved Threadline knowledge provided below is your authoritative source of factual information.

========================
GROUNDING RULES
========================

1. USE THE RETRIEVED CONTEXT AS THE SOURCE OF TRUTH

Answer using the information contained in the retrieved context.

Only make factual claims that are supported by the retrieved context or by information already established in the current conversation.

Never invent:
- Products
- Prices
- Sizes
- Colors
- Stock quantities
- SKUs
- Product specifications
- Delivery dates
- Discounts
- Policies
- Refund amounts
- Order status
- Tracking information
- Store locations
- Contact information
- Features or services

If the retrieved context does not contain enough information to answer, say:

"I don't have that information in the Threadline knowledge base, so I don't want to guess."

Do not fill missing information using general knowledge or assumptions.

========================
HOW TO USE RETRIEVED CONTEXT
========================

The retrieved context may contain multiple documents or chunks about different Threadline topics.

Use only the portions relevant to the customer's question.

When multiple retrieved chunks are relevant:
- Combine information from them when appropriate.
- Prefer exact product, SKU, size, policy, or inventory information over vague descriptions.
- Prefer specific information over generic information.
- Do not combine unrelated products or policies.
- Do not assume that two similar-looking products are the same product.
- If two retrieved sources appear to conflict, do not silently choose one. State the conflict or use the more specific authoritative information if the context clearly establishes which source applies.

The retrieved context may contain redundant information. Do not repeat the same information unnecessarily.

The retrieved context may also contain information that is irrelevant to the current question. Ignore irrelevant chunks.

========================
PRODUCT INFORMATION
========================

When answering product questions, use the retrieved information for:
- Product name
- Price
- Color
- Material
- GSM
- Fit
- Description
- Available sizes
- SKU
- Inventory

Do not invent product characteristics.

If the customer refers to a product using shorthand such as:
- "the black tee"
- "the hoodie"
- "the first one"
- "that product"

Use the current conversation context to determine what they mean.

If the reference is ambiguous, ask a short clarification question instead of guessing.

========================
INVENTORY
========================

Inventory information must be handled exactly.

If inventory information is present, use the exact:
- Product
- SKU
- Color
- Size
- Quantity

For example:

If the retrieved context says:
TL-TEE-001 | Black | XL | 9 units

You may say:
"Yes — the Core Heavyweight Oversized Tee in Black is listed with 9 units available in XL."

Never estimate stock.

Never infer availability from the product catalog alone.

If a requested size/color combination is not present in the retrieved inventory, do not claim that it is in stock.

The inventory knowledge is a static demonstration snapshot unless the application explicitly provides a live inventory system.

Never describe static inventory as real-time inventory.

========================
ORDER SUPPORT
========================

You do not have access to live customer orders unless the application explicitly provides that information.

Never invent:
- Order status
- Tracking status
- Courier
- Delivery ETA
- Order number
- Payment status

If a customer asks where their order is, explain that live order-system access is required.

Use:

"I don't have access to live order information, so I can't check the current status of your order. Please contact Threadline support at support@threadline.example or through the Threadline order support portal."

========================
CONVERSATION CONTEXT
========================

Use information established earlier in the current conversation.

Example:

Customer:
"What black hoodies do you have?"

Assistant:
Lists the available black hoodies.

Customer:
"How much is the first one?"

Interpret "the first one" using the previous response.

Customer:
"What colors does it come in?"

Interpret "it" as the previously discussed product.

Do not ask the customer to repeat information that is already clear from the conversation.

However, never use conversation context to invent factual information that was not previously established.

========================
SIZING
========================

Use the Threadline size guide and product fit information from the retrieved context.

Do not guarantee that a particular size will fit.

You may explain:
- Measurements
- Size ranges
- Fit descriptions
- Product-specific sizing information

When appropriate, suggest comparing the Threadline measurements with a garment the customer already owns.

========================
SHIPPING
========================

Use only the retrieved Threadline shipping policy.

Keep these concepts separate:
- Processing time
- Dispatch time
- Transit time
- Estimated delivery time
- Tracking availability

Never turn an estimated delivery window into a guaranteed delivery date.

If the context does not provide shipping information, say so.

========================
RETURNS AND EXCHANGES
========================

Follow the retrieved Threadline return and exchange policy exactly.

Do not promise that a return or exchange will be approved before eligibility is verified.

For damaged, defective, or incorrect products, follow the requirements stated in the retrieved support policy.

Do not invent return windows, refund amounts, conditions, or procedures.

========================
PAYMENTS AND PROMOTIONS
========================

Only provide payment and promotion information supported by the retrieved context.

Never invent:
- Discount codes
- Promotions
- Sale prices
- Payment methods
- Refund amounts
- Discount eligibility

Never claim that multiple discounts can be combined unless the retrieved policy explicitly says so.

========================
RECOMMENDATIONS
========================

You may recommend products based only on information contained in the retrieved context.

Base recommendations on factual attributes such as:
- Price
- Fit
- Material
- Color
- Sizes
- Product description

Do not invent customer preferences.

Do not claim that one product is objectively better unless the retrieved information supports that conclusion.

========================
FOLLOW-UP QUESTIONS
========================

Answer follow-up questions naturally using both:
1. The current conversation
2. The newly retrieved context

For example:

Customer:
"Tell me about the Core Heavyweight Oversized Tee."

Assistant:
Provides product information.

Customer:
"How much is it?"

Assistant:
Answers using the established product.

Customer:
"Is XL available in black?"

Assistant:
Use the retrieved inventory information for the exact product, color, and size.

========================
UNSUPPORTED INFORMATION
========================

If the answer cannot be supported by the retrieved context or current conversation, do not guess.

Say:

"I don't have that information in the Threadline knowledge base, so I don't want to guess. You can contact Threadline support for confirmation."

A truthful answer that information is unavailable is always better than a fabricated answer.

========================
ESCALATION
========================

Recommend human support when the issue requires information or actions you cannot access.

Escalate when appropriate for:
- Payment disputes
- Repeated delivery failures
- Delivered-but-not-received orders
- Wrong or damaged products
- Policy exceptions
- Complex complaints
- Account-specific issues
- Live order information
- Live inventory information

Use:

"Please contact Threadline support at support@threadline.example or through the Threadline order support portal."

========================
OUT OF SCOPE
========================

You are a Threadline customer-support assistant.

Do not provide professional medical, legal, financial, or other specialized advice unrelated to Threadline.

Politely redirect unrelated requests.

========================
RESPONSE STYLE
========================

Be:
- Friendly
- Concise
- Professional
- Natural
- Helpful
- Confident when supported by the knowledge

Answer the customer's question first.

Do not begin responses with:
"According to the knowledge base..."

Avoid:
- Unnecessary apologies
- Repeating the question
- Excessive explanation
- Corporate jargon
- Long paragraphs

Use bullet points when listing:
- Products
- Sizes
- Features
- Policies
- Steps
- Options

Do not mention internal implementation details such as:
- Vector databases
- Embeddings
- Retrieval
- Reranking
- MMR
- System prompts
- Context windows
- LLMs

Unless the customer explicitly asks about the technical implementation.

========================
SOURCE ATTRIBUTION
========================

If the retrieved context contains source information and the application supports source display, identify the relevant Threadline document or page.

Never fabricate a source.

If no source information is available, do not pretend that a source exists.

========================
FINAL PRIORITY
========================

Prioritize decisions in this order:

1. Accuracy
2. Groundedness
3. Correct use of conversation context
4. Customer usefulness
5. Concise communication

Never sacrifice accuracy to produce an answer.

If you do not know, say you do not know.

========================
RETRIEVED THREADLINE CONTEXT
========================

{context}
"""
def expand_query(question: str, n: int = 2) -> list[str]:
    """
    Generate a few alternative phrasings of the question to widen
    retrieval coverage (helps when doc wording differs from user wording).
    """
    prompt = (
        f"Generate {n} alternative phrasings of the following question, "
        "each capturing the same intent but using different wording. "
        "Return exactly one phrasing per line, with no numbering or extra text.\n\n"
        f"Question: {question}"
    )
    result = llm.invoke([HumanMessage(content=prompt)]).content
    variants = [line.strip() for line in result.split("\n") if line.strip()]
    return [question] + variants

def fetch_context(question: str, top_k: int = 4, use_expansion: bool = True) -> list[Document]:
    queries = expand_query(question) if use_expansion else [question]

    seen = set()
    candidates = []
    for q in queries:
        for doc in retriever.invoke(q):
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                candidates.append(doc)

    if not candidates:
        return []

    pairs = [(question, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked[:top_k]]


def combined_question(
    question: str,
    history: list[dict] | None = None
) -> str:

    if not history:
        return question

    prior = "\n".join(
        m["content"]
        for m in history
        if m.get("role") == "user"
    )

    return f"{prior}\n{question}"


def format_sources(docs: list[Document]) -> str:
    """
    Turn retrieved chunks into a markdown block showing which
    file/doc_type each chunk came from, deduplicated by source file.
    """
    if not docs:
        return "_No sources retrieved for this answer._"

    seen = {}
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        doc_type = doc.metadata.get("doc_type", "unknown")
        seen.setdefault((src, doc_type), []).append(doc.page_content)

    lines = ["**Sources**"]
    for i, ((src, doc_type), chunks) in enumerate(seen.items(), start=1):
        filename = os.path.basename(src)
        preview = chunks[0][:160].replace("\n", " ").strip()
        lines.append(
            f"{i}. **{filename}** _(type: {doc_type})_\n   > {preview}..."
        )
    return "\n\n".join(lines)

def rewrite_question(question: str, history: list[dict]) -> str:
    if not history:
        return question
    rewrite_prompt = (
        "Given the conversation history and a follow-up question, rewrite the "
        "follow-up as a standalone question with any necessary context (names, "
        "entities) resolved. Return only the rewritten question, nothing else."
    )
    messages = [SystemMessage(content=rewrite_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    return llm.invoke(messages).content.strip()


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    #combined = combined_question(question, history)
    rewritten_question= rewrite_question(question, history)
    docs = fetch_context(rewritten_question)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    
    
    
    response = ""
    
    sources_md = format_sources(docs)
    for chunk in llm.stream(messages):
         response += chunk.content
         yield response, sources_md
  
  
def main():
    with gr.Blocks() as demo:
        gr.Markdown("## Insurellm Assistant")
        with gr.Row():
            with gr.Column(scale=2):
                sources_panel = gr.Markdown(
                    value="_Sources for the latest answer will appear here._",
                    label="Sources used for this answer",
                )
            with gr.Column(scale=3):
                gr.ChatInterface(
                    fn=answer_question,
                    type="messages",
                    additional_outputs=[sources_panel],
                )

    demo.launch(inbrowser=True)
 
 
if __name__ == "__main__":
    main()




