from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load Sentence Transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_pdf_text(pdf_file):
    """
    Extract text from an uploaded PDF.
    """

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def split_text(text, chunk_size=500):
    """
    Split large text into smaller chunks.
    """

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        if chunk.strip():
            chunks.append(chunk)

    return chunks


def create_embeddings(chunks):
    """
    Convert text chunks into numerical embeddings.
    """

    if not chunks:
        return []

    embeddings = embedding_model.encode(chunks)

    return embeddings


def find_relevant_chunks(question, chunks, embeddings, top_k=3):
    """
    Find the most relevant PDF chunks for a question.
    """

    if not chunks or len(embeddings) == 0:
        return []

    question_embedding = embedding_model.encode(
        [question]
    )

    similarity_scores = cosine_similarity(
        question_embedding,
        embeddings
    )[0]

    top_indices = similarity_scores.argsort()[-top_k:][::-1]

    relevant_chunks = []

    for index in top_indices:
        relevant_chunks.append(chunks[index])

    return relevant_chunks


def create_rag_prompt(question, relevant_chunks):
    """
    Create a prompt for Ollama using retrieved PDF information.
    """

    context = "\n\n".join(relevant_chunks)

    prompt = f"""
You are an AI Smart Farming Assistant.

Answer the farmer's question using the agricultural
information provided below.

IMPORTANT:
- Give a simple and clear answer.
- Use only the useful information from the context.
- Do not invent information that is not supported by the context.
- If the context does not contain enough information,
  clearly say that the uploaded document does not contain
  enough information.

Agricultural Information:
{context}

Farmer's Question:
{question}

Answer:
"""

    return prompt