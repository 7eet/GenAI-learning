import os
import streamlit as st

from dotenv import load_dotenv

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from sentence_transformers import CrossEncoder

from rank_bm25 import BM25Okapi

from src.ingestion.document_loader import (
    load_document,
    split_documents
)


# =========================================================
# Configuration
# =========================================================

load_dotenv()

st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# Cached Models
# =========================================================

@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2"
    )


@st.cache_resource
def get_reranker():
    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )


embeddings = get_embeddings()
reranker = get_reranker()
llm = get_llm()


# =========================================================
# Reranking
# =========================================================

def rerank_documents(
    query,
    documents,
    top_k=3
):
    """
    Rerank retrieved candidate documents
    using a cross-encoder.
    """

    pairs = [
        (query, doc.page_content)
        for doc in documents
    ]

    scores = reranker.predict(pairs)

    ranked_documents = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked_documents[:top_k]


# =========================================================
# BM25 Keyword Search
# =========================================================

def keyword_search(
    query,
    chunks,
    bm25,
    top_k=5
):
    """
    Perform keyword-based retrieval using BM25.
    """

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(
        tokenized_query
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    return [
        (chunks[i], scores[i])
        for i in ranked_indices
    ]


# =========================================================
# Reciprocal Rank Fusion
# =========================================================

def reciprocal_rank_fusion(
    semantic_results,
    keyword_results,
    k=60
):
    """
    Combine semantic and keyword rankings
    using Reciprocal Rank Fusion (RRF).

    RRF uses document rank rather than
    combining raw retrieval scores.
    """

    scores = {}
    documents = {}

    # -----------------------------
    # Semantic results
    # -----------------------------

    for rank, doc in enumerate(
        semantic_results,
        start=1
    ):

        doc_id = doc.page_content

        documents[doc_id] = doc

        scores[doc_id] = (
            scores.get(doc_id, 0)
            + 1 / (k + rank)
        )

    # -----------------------------
    # Keyword results
    # -----------------------------

    for rank, (doc, _) in enumerate(
        keyword_results,
        start=1
    ):

        doc_id = doc.page_content

        documents[doc_id] = doc

        scores[doc_id] = (
            scores.get(doc_id, 0)
            + 1 / (k + rank)
        )

    # -----------------------------
    # Final ranking
    # -----------------------------

    ranked_results = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        (documents[doc_id], score)
        for doc_id, score in ranked_results
    ]


# =========================================================
# Document Formatting
# =========================================================

def format_docs(documents):
    """
    Convert retrieved Documents into
    context for the LLM.
    """

    return "\n\n".join(
        doc.page_content
        for doc in documents
    )


# =========================================================
# RAG Prompt
# =========================================================

rag_prompt = ChatPromptTemplate.from_template(
    """
You are an AI Knowledge Assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say:

"I don't have enough information in the provided documents."

Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
"""
)


# =========================================================
# RAG Chain
# =========================================================

rag_chain = (
    rag_prompt
    | llm
    | StrOutputParser()
)


# =========================================================
# UI
# =========================================================

st.title("🤖 AI Knowledge Assistant")

    # st.markdown(
    #     """
    # Upload a document and ask questions about it.

    # The application demonstrates multiple RAG retrieval
    # strategies and shows how the retrieved results change
    # through each stage of the pipeline.
    # """
    # )


# =========================================================
# File Upload
# =========================================================

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "txt", "md"]
)


if uploaded_file:

    # =====================================================
    # Save Uploaded File
    # =====================================================

    os.makedirs(
        "data",
        exist_ok=True
    )

    file_path = os.path.join(
        "data",
        uploaded_file.name
    )

    with open(
        file_path,
        "wb"
    ) as f:
        f.write(
            uploaded_file.getvalue()
        )


    # =====================================================
    # Document Loading
    # =====================================================

    documents = load_document(
        file_path
    )

    st.success(
        f"Loaded {len(documents)} document(s)"
    )


    # =====================================================
    # Chunking
    # =====================================================

    chunks = split_documents(
        documents
    )

    st.success(
        f"Created {len(chunks)} chunks"
    )


    # =====================================================
    # Document Information
    # =====================================================

    with st.expander("📄 Document Information"):

        st.write(
            f"Original documents: {len(documents)}"
        )

        st.write(
            f"Total chunks: {len(chunks)}"
        )

        if chunks:

            st.write(
                "Example chunk metadata:"
            )

            st.write(
                chunks[0].metadata
            )


    # =====================================================
    # BM25 Index
    # =====================================================

    tokenized_chunks = [
        chunk.page_content.lower().split()
        for chunk in chunks
    ]

    bm25 = BM25Okapi(
        tokenized_chunks
    )


    # =====================================================
    # FAISS Vector Store
    # =====================================================

    with st.spinner(
        "Creating FAISS vector store..."
    ):

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

    st.success(
        "FAISS vector store created successfully"
    )


    # =====================================================
    # Question Input
    # =====================================================

    st.subheader("💬 Ask a Question")

    query = st.text_area(
        "Enter your question",
        placeholder=(
            "Example: How many annual leave days "
            "do employees get?"
        ),
        height=100
    )

    ask_button = st.button(
        "🔍 Ask",
        type="primary"
    )


    # =====================================================
    # Execute RAG Pipeline
    # =====================================================

    if ask_button:

        if not query.strip():

            st.warning(
                "Please enter a question."
            )

            st.stop()


        # =================================================
        # 1. Semantic Search
        # =================================================

        with st.spinner(
            "Running semantic search..."
        ):

            semantic_results = (
                vectorstore.similarity_search_with_score(
                    query,
                    k=5
                )
            )


        with st.expander(
            "🔎 1. Semantic Search — Top 5",
            expanded=True
        ):

            st.caption(
                "FAISS semantic retrieval. "
                "Lower distance indicates greater similarity "
                "for this FAISS configuration."
            )

            for i, (doc, score) in enumerate(
                semantic_results,
                start=1
            ):

                st.markdown(
                    f"### Rank {i}"
                )

                st.write(
                    f"**Distance Score:** {score:.4f}"
                )

                st.write(
                    doc.page_content
                )

                st.write(
                    "**Metadata:**",
                    doc.metadata
                )

                st.divider()


        # =================================================
        # 2. MMR Retrieval
        # =================================================

        mmr_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 3,
                "fetch_k": 8,
                "lambda_mult": 0.9
            }
        )


        with st.spinner(
            "Running MMR retrieval..."
        ):

            mmr_results = mmr_retriever.invoke(
                query
            )


        with st.expander(
            "🎯 2. MMR Retrieval — Top 3",
            expanded=False
        ):

            st.caption(
                "MMR balances relevance and diversity. "
                "The retriever considers 8 candidates "
                "and returns 3 documents."
            )

            for i, doc in enumerate(
                mmr_results,
                start=1
            ):

                st.markdown(
                    f"### Rank {i}"
                )

                st.write(
                    doc.page_content
                )

                st.write(
                    "**Metadata:**",
                    doc.metadata
                )

                st.divider()


        # =================================================
        # 3. BM25 Keyword Search
        # =================================================

        with st.spinner(
            "Running BM25 keyword search..."
        ):

            keyword_results = keyword_search(
                query,
                chunks,
                bm25,
                top_k=5
            )


        with st.expander(
            "🔤 3. BM25 Keyword Search — Top 5",
            expanded=False
        ):

            st.caption(
                "Keyword-based retrieval using BM25. "
                "Higher BM25 scores indicate stronger "
                "keyword matching."
            )

            for i, (doc, score) in enumerate(
                keyword_results,
                start=1
            ):

                st.markdown(
                    f"### Rank {i}"
                )

                st.write(
                    f"**BM25 Score:** {score:.4f}"
                )

                st.write(
                    doc.page_content
                )

                st.divider()


        # =================================================
        # 4. Hybrid Search
        # =================================================

        semantic_documents = [
            doc
            for doc, _ in semantic_results
        ]


        hybrid_results = (
            reciprocal_rank_fusion(
                semantic_documents,
                keyword_results
            )
        )


        with st.expander(
            "🔀 4. Hybrid Search — Semantic + BM25 + RRF",
            expanded=False
        ):

            st.caption(
                "RRF combines the rankings from semantic "
                "and keyword retrieval."
            )

            for i, (doc, score) in enumerate(
                hybrid_results[:5],
                start=1
            ):

                st.markdown(
                    f"### Rank {i}"
                )

                st.write(
                    f"**RRF Score:** {score:.6f}"
                )

                st.write(
                    doc.page_content
                )

                st.write(
                    "**Metadata:**",
                    doc.metadata
                )

                st.divider()


        # =================================================
        # 5. Candidate Selection
        # =================================================

        candidates = [
            doc
            for doc, _ in hybrid_results[:8]
        ]


        # =================================================
        # 6. Cross-Encoder Reranking
        # =================================================

        with st.spinner(
            "Reranking retrieved candidates..."
        ):

            ranked_results = rerank_documents(
                query,
                candidates,
                top_k=3
            )


        with st.expander(
            "🎯 5. Cross-Encoder Reranking — Top 3",
            expanded=True
        ):

            st.caption(
                "The cross-encoder evaluates the query "
                "and candidate document together and "
                "reranks the candidates by relevance."
            )

            for i, (doc, score) in enumerate(
                ranked_results,
                start=1
            ):

                st.markdown(
                    f"### Rank {i}"
                )

                st.write(
                    f"**Reranker Score:** {score:.4f}"
                )

                st.write(
                    doc.page_content
                )

                st.write(
                    "**Metadata:**",
                    doc.metadata
                )

                st.divider()


        # =================================================
        # 7. Final Context
        # =================================================

        final_documents = [
            doc
            for doc, _ in ranked_results
        ]

        context = format_docs(
            final_documents
        )


        # =================================================
        # 8. Grounded Generation
        # =================================================

        with st.spinner(
            "Generating grounded answer..."
        ):

            answer = rag_chain.invoke(
                {
                    "context": context,
                    "question": query
                }
            )


        # =================================================
        # 9. Final Answer
        # =================================================

        st.subheader(
            "💬 Final Answer"
        )

        st.success(
            answer
        )


        # =================================================
        # 10. Sources
        # =================================================

        st.subheader(
            "📚 Sources"
        )

        for i, (doc, score) in enumerate(
            ranked_results,
            start=1
        ):

            source = doc.metadata.get(
                "source",
                "Unknown source"
            )

            page = doc.metadata.get(
                "page_label",
                doc.metadata.get(
                    "page",
                    "Unknown"
                )
            )

            st.write(
                f"**Source {i}:** {source}"
            )

            st.write(
                f"**Page:** {page}"
            )

            st.write(
                f"**Reranker Score:** {score:.4f}"
            )

            st.divider()