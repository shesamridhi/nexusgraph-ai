"""
NexusGraph AI — Sample Data Seeder
Pre-populates the knowledge graph with sample AI/ML research papers
so the system works out-of-the-box for demos.
"""
import structlog
from db.neo4j_client import neo4j_client
from db.chroma_client import chroma_client

logger = structlog.get_logger(__name__)

SAMPLE_PAPERS = [
    {
        "doc_id": "paper_attention_001",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit"],
        "topics": ["Transformers", "Self-Attention", "NLP", "Deep Learning", "Neural Machine Translation"],
        "content": """Attention Is All You Need
Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit (Google Brain / Google Research)

Abstract:
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks 
that include an encoder and a decoder. The best performing models also connect the encoder and decoder 
through an attention mechanism. We propose a new simple network architecture, the Transformer, based 
solely on attention mechanisms, dispensing with recurrence and convolutions entirely.

The Transformer model achieves superior results on machine translation tasks. On the WMT 2014 English-to-German 
translation task, the Transformer achieves 28.4 BLEU, improving over the existing best results. 
On WMT 2014 English-to-French, it achieves 41.0 BLEU.

The self-attention mechanism allows the model to attend to all positions in the input sequence simultaneously.
Multi-head attention runs through the attention function in parallel, with different learned linear projections.
The positional encoding adds information about the position of tokens in the sequence.

Key contributions:
- Transformer architecture eliminates recurrence entirely
- Multi-head self-attention captures long-range dependencies efficiently
- Positional encoding preserves sequence order information
- Layer normalization stabilizes training
- Feed-forward sublayers add non-linear transformations
"""
    },
    {
        "doc_id": "paper_bert_002",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "authors": ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
        "topics": ["BERT", "Pre-training", "NLP", "Transformers", "Language Models", "Fine-tuning"],
        "content": """BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
Authors: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (Google AI Language)

Abstract:
We introduce a new language representation model called BERT, which stands for Bidirectional Encoder 
Representations from Transformers. BERT is designed to pre-train deep bidirectional representations from 
unlabeled text by jointly conditioning on both left and right context in all layers.

BERT uses two pre-training objectives:
1. Masked Language Model (MLM): randomly masks tokens and predicts them
2. Next Sentence Prediction (NSP): predicts if one sentence follows another

The pre-trained BERT model can be fine-tuned with just one additional output layer for many downstream 
tasks, including question answering, language inference, and named entity recognition.

BERT achieves state-of-the-art results on eleven NLP tasks:
- GLUE score: 80.5% (7.7% improvement)
- MultiNLI: 86.7% accuracy  
- SQuAD v1.1 F1: 93.2
- SQuAD v2.0 F1: 83.1

The bidirectional nature of BERT allows it to understand context from both directions simultaneously,
unlike previous models like GPT which only condition on left context.
"""
    },
    {
        "doc_id": "paper_rag_003",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": ["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus", "Fabio Petroni"],
        "topics": ["RAG", "Retrieval-Augmented Generation", "NLP", "Knowledge Retrieval", "Open-Domain QA"],
        "content": """Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
Authors: Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni (Facebook AI Research / University College London)

Abstract:
Large pre-trained language models have been shown to store factual knowledge in their parameters.
However, this knowledge is static and limited. We propose RAG — Retrieval-Augmented Generation — 
a general-purpose fine-tuning approach for retrieval-augmented language models.

RAG combines a pre-trained retriever with a pre-trained seq2seq model:
1. The retriever finds relevant documents from a non-parametric memory (e.g., Wikipedia)
2. The generator conditions on retrieved documents to produce the final answer

RAG models outperform parametric seq2seq models on knowledge-intensive tasks:
- TriviaQA: 56.8% exact match
- Natural Questions: 44.5% exact match  
- WebQuestions: 45.5% exact match

The key advantage of RAG is that the knowledge can be updated without retraining — 
just update the document index. This is crucial for real-world applications requiring 
up-to-date information.

Applications include open-domain QA, fact verification, slot filling, and document summarization.
"""
    },
    {
        "doc_id": "paper_gnn_004",
        "title": "Graph Neural Networks: A Review of Methods and Applications",
        "authors": ["Jie Zhou", "Ganqu Cui", "Shengding Hu", "Zhengyan Zhang"],
        "topics": ["Graph Neural Networks", "GNN", "Deep Learning", "Graph Learning", "Node Classification"],
        "content": """Graph Neural Networks: A Review of Methods and Applications
Authors: Jie Zhou, Ganqu Cui, Shengding Hu, Zhengyan Zhang (Tsinghua University)

Abstract:
Graph neural networks (GNNs) are a class of deep learning methods designed to perform inference 
on data described by graphs. GNNs are neural networks that can be directly applied to graphs, 
providing a convenient framework for node-level, edge-level, and graph-level prediction tasks.

Core GNN architectures:
1. Graph Convolutional Networks (GCN): aggregate neighbor features using normalized adjacency
2. Graph Attention Networks (GAT): use attention weights for neighbor aggregation
3. GraphSAGE: sample and aggregate from local neighborhoods
4. Message Passing Neural Networks (MPNN): generalized framework

Applications:
- Citation networks: node classification for paper categorization
- Social networks: community detection and link prediction
- Knowledge graphs: entity and relation prediction
- Drug discovery: molecular property prediction
- Recommendation systems: user-item interaction graphs

GNNs address the fundamental limitation of CNNs and RNNs by handling non-Euclidean data structures.
The message passing paradigm allows nodes to iteratively aggregate information from their neighbors,
building up rich representations of local and global graph structure.
"""
    },
    {
        "doc_id": "paper_langchain_005",
        "title": "LangChain: Building Applications with Large Language Models",
        "authors": ["Harrison Chase", "Ankush Gola"],
        "topics": ["LangChain", "LLM", "Agents", "Chains", "RAG", "Tool Use", "AI Applications"],
        "content": """LangChain: Building Applications with Large Language Models
Authors: Harrison Chase, Ankush Gola (LangChain Inc.)

Overview:
LangChain is an open-source framework for developing applications powered by language models.
It provides abstractions for chaining LLM calls, integrating external tools, and building 
autonomous agents.

Core components:
1. Models: wrappers for LLMs (OpenAI, Anthropic, HuggingFace, etc.)
2. Prompts: prompt templates and example selectors
3. Chains: sequences of LLM calls and transformations
4. Agents: LLMs that decide which tools to use based on input
5. Memory: persistence of state across chain/agent calls
6. Indexes: structured access to documents (vector stores, document loaders)

Key integrations:
- Vector databases: Chroma, Pinecone, Weaviate, FAISS
- Document loaders: PDF, HTML, CSV, Notion, GitHub
- Tools: Google Search, Python REPL, Wikipedia, SQL
- LLM providers: OpenAI GPT-4, Anthropic Claude, Google Gemini

LangChain enables complex applications like:
- Document Q&A systems with RAG
- Code generation and execution agents
- Multi-step research pipelines
- Conversational chatbots with memory

The framework abstracts the complexity of LLM orchestration while remaining flexible enough 
for production deployments.
"""
    },
    {
        "doc_id": "paper_graph_rag_006",
        "title": "From Local to Global: A Graph RAG Approach to Query-Focused Summarization",
        "authors": ["Edge Darren", "Ha Trinh", "Newman Newman", "Julie Kim"],
        "topics": ["GraphRAG", "Knowledge Graph", "RAG", "Summarization", "Community Detection", "NLP"],
        "content": """From Local to Global: A Graph RAG Approach to Query-Focused Summarization
Authors: Darren Edge, Ha Trinh, Newman Newman, Julie Kim (Microsoft Research)

Abstract:
The use of retrieval-augmented generation (RAG) to retrieve relevant information from an external 
knowledge source enables large language models (LLMs) to answer questions about private and/or 
previously unseen document collections. However, RAG fails on global questions directed at an 
entire text corpus, such as "What are the main themes in the dataset?"

We propose Graph RAG, an approach that uses knowledge graph memory structures to improve:
1. Question answering on private document collections
2. Query-focused summarization across entire corpora
3. Multi-hop reasoning about connected information

Graph RAG pipeline:
- Build entity-relationship graph from documents
- Detect communities using hierarchical clustering
- Pre-summarize communities at multiple granularities  
- At query time: retrieve relevant community summaries

Results show that Graph RAG reduces hallucination and improves comprehensiveness on 
global sensemaking tasks compared to baseline RAG approaches.

Applications: investigative journalism, legal case analysis, financial report analysis,
academic literature review, enterprise knowledge management.
"""
    },
]


async def seed_sample_data():
    """Insert sample papers if the database is empty."""
    # Check if already seeded
    stats = await neo4j_client.get_stats()
    if stats["total_nodes"] > 10:
        logger.info("seed.skip", reason="Data already exists")
        return

    logger.info("seed.start", papers=len(SAMPLE_PAPERS))

    for paper in SAMPLE_PAPERS:
        try:
            # Vector store
            await chroma_client.ingest_document(
                doc_id=paper["doc_id"],
                content=paper["content"],
                metadata={
                    "title": paper["title"],
                    "authors": str(paper["authors"]),
                    "topics": str(paper["topics"]),
                    "source": "seed_data",
                }
            )

            # Graph: Document node
            await neo4j_client.upsert_document(
                doc_id=paper["doc_id"],
                title=paper["title"],
                content_preview=paper["content"][:300],
                metadata={"source": "seed_data"},
            )

            # Graph: Author nodes + WROTE edges
            for author in paper["authors"]:
                await neo4j_client.upsert_author(name=author)
                await neo4j_client.create_relationship(
                    from_id=author, from_label="Author",
                    to_id=paper["doc_id"], to_label="Document",
                    rel_type="WROTE",
                )

            # Graph: Co-author edges
            authors = paper["authors"]
            for i in range(len(authors)):
                for j in range(i + 1, len(authors)):
                    await neo4j_client.create_relationship(
                        from_id=authors[i], from_label="Author",
                        to_id=authors[j], to_label="Author",
                        rel_type="CO_AUTHORED_WITH",
                    )

            # Graph: Topic nodes + COVERS edges
            for topic in paper["topics"]:
                await neo4j_client.upsert_topic(name=topic, category="ML/AI")
                await neo4j_client.create_relationship(
                    from_id=topic, from_label="Topic",
                    to_id=paper["doc_id"], to_label="Document",
                    rel_type="COVERS",
                )

        except Exception as e:
            logger.warning("seed.paper_failed", title=paper["title"][:40], error=str(e))

    logger.info("seed.complete", papers=len(SAMPLE_PAPERS))
