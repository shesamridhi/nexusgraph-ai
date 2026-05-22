"""
NexusGraph AI — Sample Data Seeder
Pre-populates ChromaDB with sample AI/ML research papers for demo.
"""
import structlog
from db.chroma_client import chroma_client

logger = structlog.get_logger(__name__)

SAMPLE_PAPERS = [
    {
        "doc_id": "paper_attention_001",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit"],
        "topics": ["Transformers", "Self-Attention", "NLP", "Deep Learning"],
        "content": """Attention Is All You Need
Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit (Google Brain)

The Transformer architecture relies entirely on self-attention mechanisms to draw global dependencies between input and output. Unlike RNNs and CNNs, Transformers process all tokens in parallel using multi-head attention. The encoder-decoder structure uses positional encodings to retain sequence order.

Self-attention computes query, key, and value vectors. The dot product of query and key gives attention weights via softmax. These weights compute a weighted sum of values. Multi-head attention runs this in parallel with different learned projections.

Key contributions:
- Transformer architecture eliminates recurrence entirely
- Multi-head self-attention captures long-range dependencies efficiently  
- Positional encoding preserves sequence order information
- Achieves 28.4 BLEU on WMT 2014 English-to-German translation
- Foundation for BERT, GPT, and all modern LLMs
"""
    },
    {
        "doc_id": "paper_bert_002",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "authors": ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
        "topics": ["BERT", "Pre-training", "NLP", "Transformers", "Language Models"],
        "content": """BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
Authors: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (Google AI Language)

BERT stands for Bidirectional Encoder Representations from Transformers. Unlike GPT which is unidirectional, BERT reads text bidirectionally — conditioning on both left and right context simultaneously.

Two pre-training objectives:
1. Masked Language Model (MLM): randomly masks 15% of tokens and predicts them
2. Next Sentence Prediction (NSP): predicts if sentence B follows sentence A

BERT achieves state-of-the-art on 11 NLP tasks:
- GLUE score: 80.5% (7.7% absolute improvement)
- SQuAD v1.1 F1: 93.2
- MultiNLI accuracy: 86.7%

Fine-tuning BERT with one output layer achieves SOTA on QA, NER, and classification tasks.
"""
    },
    {
        "doc_id": "paper_rag_003",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": ["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus", "Fabio Petroni"],
        "topics": ["RAG", "Retrieval-Augmented Generation", "NLP", "Knowledge Retrieval"],
        "content": """Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
Authors: Patrick Lewis, Ethan Perez (Facebook AI Research)

RAG combines a pre-trained retriever with a pre-trained seq2seq generator:
1. Retriever finds relevant documents from a non-parametric memory (Wikipedia DPR index)
2. Generator conditions on retrieved documents to produce the final answer

Key advantage: knowledge can be updated without retraining — just update the document index.

Results on knowledge-intensive tasks:
- TriviaQA: 56.8% exact match
- Natural Questions: 44.5% exact match
- WebQuestions: 45.5% exact match

RAG outperforms parametric-only models and reduces hallucination by grounding answers in retrieved evidence. Applications include open-domain QA, fact verification, slot filling, and summarization.
"""
    },
    {
        "doc_id": "paper_gnn_004",
        "title": "Graph Neural Networks: A Review of Methods and Applications",
        "authors": ["Jie Zhou", "Ganqu Cui", "Shengding Hu", "Zhengyan Zhang"],
        "topics": ["Graph Neural Networks", "GNN", "Deep Learning", "Node Classification"],
        "content": """Graph Neural Networks: A Review of Methods and Applications
Authors: Jie Zhou, Ganqu Cui (Tsinghua University)

GNNs are deep learning methods designed for graph-structured data. Core architectures:
1. Graph Convolutional Networks (GCN): aggregate neighbor features using normalized adjacency
2. Graph Attention Networks (GAT): use attention weights for neighbor aggregation
3. GraphSAGE: sample and aggregate from local neighborhoods
4. Message Passing Neural Networks (MPNN): generalized message passing framework

Applications:
- Citation networks: node classification for paper categorization
- Social networks: community detection and link prediction
- Knowledge graphs: entity and relation prediction
- Drug discovery: molecular property prediction
- Recommendation systems: user-item interaction graphs

GNNs address the limitation of CNNs by handling non-Euclidean graph data structures.
"""
    },
    {
        "doc_id": "paper_graphrag_005",
        "title": "Graph RAG: Graph-Augmented Retrieval for Query-Focused Summarization",
        "authors": ["Darren Edge", "Ha Trinh", "Newman Newman", "Julie Kim"],
        "topics": ["GraphRAG", "Knowledge Graph", "RAG", "Summarization", "Microsoft Research"],
        "content": """From Local to Global: A Graph RAG Approach to Query-Focused Summarization
Authors: Darren Edge, Ha Trinh, Julie Kim (Microsoft Research)

Graph RAG uses knowledge graph memory structures to improve RAG for global questions like "What are the main themes in this dataset?" — which baseline RAG fails at.

Graph RAG pipeline:
1. Build entity-relationship graph from documents using LLM extraction
2. Detect communities using hierarchical Leiden clustering
3. Pre-summarize communities at multiple granularities
4. At query time: retrieve relevant community summaries + local context

Results: Graph RAG reduces hallucination and improves comprehensiveness on global sensemaking tasks compared to naive RAG. Particularly effective for investigative journalism, legal analysis, and enterprise knowledge management.
"""
    },
    {
        "doc_id": "paper_langgraph_006",
        "title": "LangGraph: Multi-Agent Orchestration with Cyclic Graphs",
        "authors": ["Harrison Chase", "Ankush Gola"],
        "topics": ["LangGraph", "LLM Agents", "Multi-Agent", "RAG", "LangChain"],
        "content": """LangGraph: Multi-Agent Orchestration with Cyclic Graphs
Authors: Harrison Chase, Ankush Gola (LangChain Inc.)

LangGraph extends LangChain with graph-based agent orchestration supporting cycles, branching, and multi-agent coordination. Unlike linear chains, LangGraph enables think-check-execute loops where agents can retry and revise.

Core concepts:
- StateGraph: nodes are agent functions, edges are transitions
- Conditional edges: route based on agent output (e.g., retry if confidence low)
- Checkpointing: persist state for human-in-the-loop workflows
- Multi-agent: specialized agents (Router, Researcher, Critic, Orchestrator)

NexusGraph AI uses LangGraph to implement:
1. Router Agent: classifies query as vector/graph/hybrid
2. Researcher Agent: fetches from ChromaDB + Neo4j simultaneously
3. Critic Agent: validates answer confidence, triggers retry if needed
4. Orchestrator: synthesizes final grounded answer with citations

BM25 hybrid search combines sparse keyword matching with dense vector similarity for better retrieval. Mean latency: 13.5s, Mean confidence: 0.67, Vector pass rate: 100%.
"""
    },
]


async def seed_sample_data():
    """Insert sample papers into ChromaDB on every startup (EphemeralClient resets on restart)."""
    logger.info("seed.start", papers=len(SAMPLE_PAPERS))

    for paper in SAMPLE_PAPERS:
        try:
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
            logger.info("seed.paper_ok", title=paper["title"][:40])
        except Exception as e:
            logger.warning("seed.paper_failed", title=paper["title"][:40], error=str(e))

    logger.info("seed.complete", papers=len(SAMPLE_PAPERS))