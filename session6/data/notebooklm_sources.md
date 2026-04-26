# Session 6 NotebookLM Sources — Ready to Paste

Instructions: Create 6 notebooks in NotebookLM (notebooklm.google.com), then paste each source text below.

---

## Notebook 1: CCI Session 6 - Lesson 1: Basic RAG with LlamaParse

Retrieval Augmented Generation (RAG) addresses two limitations of LLMs: finite context windows and frozen training knowledge. Instead of expecting models to know everything, RAG fetches relevant information at query time and injects it into the reasoning process. The five-stage pipeline includes loaders that bring data from PDFs and other sources, splitters that break documents into manageable chunks, embedding models that convert chunks into semantic vectors, vector stores that index and persist vectors, and retrievers that fetch relevant documents per query. For complex clinical PDFs like the National Wilms Tumor treatment guidelines with figures, tables, and multi-column layouts, naive parsers like PyPDF lose critical structure. LlamaParse uses multimodal AI to convert messy PDFs into clean markdown preserving tables as markdown tables and capturing figure descriptions. Chunking strategies range from fixed-size to semantic to hierarchical. OpenAI text-embedding-3-small offers strong general performance for clinical text. ChromaDB is a lightweight vector store that runs in Colab. The full pipeline parses the WT.pdf, chunks the markdown, embeds the chunks, stores them, then a query embeds and retrieves the top-k similar chunks for the LLM to answer over.

---

## Notebook 2: CCI Session 6 - Lesson 2: RAG Evaluation with DeepEval

RAG is easy to build poorly. The model gives confident answers based on wrong chunks, which is dangerous in clinical AI where wrong answers can harm patients. DeepEval provides four core RAG metrics that together diagnose any RAG system. Faithfulness measures whether the generated answer sticks to the retrieved context or hallucinates beyond it. Answer Relevancy measures whether the answer actually addresses the user question. Contextual Relevancy measures whether the retrieved chunks are relevant to the query. Contextual Recall measures whether all the information needed to answer was retrieved. All four use LLM-as-judge under the hood. Building a clinical test set means creating ground-truth Q&A pairs from the WT.pdf with expected answers and the actual chunks that contain those answers. Comparing a "bad" RAG (tiny 100-token chunks, no reranking) versus a "good" RAG (1000-token semantic chunks with metadata) on the same test set reveals where each metric fails. Common RAG failure modes include stale indexes, poor chunking, weak ranking, missing citations, and silent retrieval misses that produce confident wrong answers.

---

## Notebook 3: CCI Session 6 - Lesson 3: Agentic RAG with LangGraph

Basic RAG retrieves once and answers, but complex clinical questions need more than a single pass. Agentic RAG uses an agent (built with LangGraph from Session 5) that can decide when to retrieve, what to retrieve, and whether to retrieve again. Key patterns include query rewriting (refining vague clinical questions before retrieval), self-RAG (the agent grades whether retrieved documents are relevant), corrective RAG or CRAG (re-retrieval with different queries when results are insufficient), and multi-step decomposition (breaking complex questions into sub-queries). The LangGraph workflow defines a state with question, rewritten_question, documents, grade, and answer, then nodes for rewrite_query, retrieve, grade_documents, generate_answer, and re_retrieve. Conditional edges route based on the grading result: relevant chunks go to answer generation, irrelevant chunks trigger re-retrieval up to a max iteration limit. On hard clinical questions about Wilms tumor staging across treatment regimens, agentic RAG dramatically outperforms basic RAG on DeepEval metrics, especially Contextual Recall.

---

## Notebook 4: CCI Session 6 - Lesson 4: GraphRAG — Knowledge Graphs

Vector RAG is great for direct factual queries like "what is the dose of cisplatin" but fails for relational and multi-hop queries like "what are the relationships between staging, histology, and treatment outcomes." GraphRAG addresses this by extracting entities and relationships from text, building a knowledge graph, then answering via graph traversal. The architecture uses an LLM with structured output to extract entities (Disease, Stage, Drug, SideEffect, Procedure, Histology, Outcome) and relationships (TREATS, CAUSES, INDICATES, REQUIRES, ASSOCIATED_WITH) from each chunk. NetworkX provides in-memory graph storage suitable for teaching, while Neo4j scales for production. Local queries traverse from a starting entity (Stage III) through edges to find connected information (drugs that treat it, then their side effects). Global queries use community detection (Louvain or label propagation) to find themes across the graph. GraphRAG shines on multi-hop questions but is expensive to build because every chunk requires LLM extraction. The sweet spot is corpora with rich entity structure like clinical guidelines, legal documents, or research literature.

---

## Notebook 5: CCI Session 6 - Lesson 5: The Wiki Approach (Karpathy)

Andrej Karpathy proposed a paradigm shift from vector RAG: instead of retrieving raw chunks per query, maintain a living wiki of markdown files that the LLM updates incrementally. The structure includes index.md as the master table of contents, log.md tracking ingestion history, SCHEMA.md with maintainer instructions, and entity pages organized by category (entities/diseases, entities/drugs, entities/stages). When new content is ingested, the LLM identifies which entity pages should be updated, generates updated markdown content with cross-references, and updates the index. Querying skips embeddings entirely: the LLM reads index.md, identifies relevant pages, reads only those pages, and generates a citation-rich answer. Compared to vector RAG, wiki advantages include determinism (same query gives same answer), debuggability (humans can read the knowledge), no embedding drift, and built-in cross-references. Disadvantages include scaling poorly to millions of documents and requiring careful ingestion prompts. For domain-specific knowledge like KHCC pediatric oncology, the wiki could become a living physician-AI collaboration.

---

## Notebook 6: CCI Session 6 - RAG & Clinical Document Retrieval Review

Session 6 progressed through five distinct RAG approaches. Lesson 1 built basic RAG on the National Wilms Tumor guidelines using LlamaParse for complex PDF parsing, semantic chunking, OpenAI embeddings, ChromaDB vector storage, and a retrieval QA chain. Lesson 2 introduced rigorous evaluation with DeepEval's four core metrics: Faithfulness, Answer Relevancy, Contextual Relevancy, and Contextual Recall. Building a bad RAG and a good RAG side-by-side made the metrics tangible. Lesson 3 layered LangGraph agentic capabilities on top of basic RAG: query rewriting, document grading, and corrective re-retrieval, dramatically improving multi-hop question performance. Lesson 4 explored GraphRAG by extracting clinical entities and relationships into a NetworkX knowledge graph, enabling multi-hop traversal queries that vector RAG cannot answer. Lesson 5 presented Karpathy's wiki approach as a paradigm shift: pre-compiled markdown knowledge maintained by an LLM, queried by index navigation rather than embedding similarity. The session built a complete toolkit for clinical document retrieval, from foundational vector RAG to cutting-edge alternatives, with rigorous evaluation throughout.
