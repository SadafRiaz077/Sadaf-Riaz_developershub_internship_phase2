# Task 4: Context-Aware Chatbot Using LangChain (RAG)

**DevelopersHub Corporation — AI/ML Engineering Advanced Internship**

## Objective

Build a conversational chatbot that can remember context across multiple turns and retrieve external information from a custom knowledge base during conversations, using Retrieval-Augmented Generation (RAG) instead of relying solely on the language model's built-in knowledge.

## Problem Statement

Large Language Models (LLMs) generate responses based only on the knowledge encoded in their parameters at training time. This means they cannot answer questions about private, custom, or up-to-date information they were never trained on, and they are prone to hallucination when asked about niche or unfamiliar topics. This project solves that problem by combining a retrieval system with an LLM, so answers are grounded in an external, custom document set, and by adding conversational memory so the chatbot correctly handles follow-up questions.

## Dataset

A custom text knowledge base, since the task explicitly allows a custom corpus (Wikipedia pages, internal documents, or any knowledge base). The built-in sample corpus contains four short documents:

- `company_overview.txt` — background on Ezitech Software House
- `rag_systems.txt` — explanation of Retrieval-Augmented Generation
- `internship_tasks.txt` — details of the Advanced Internship Task Set
- `conversational_memory.txt` — explanation of conversational memory in LangChain

The application also supports uploading custom `.txt` files at runtime (via the Streamlit sidebar) to rebuild the knowledge base with any other document set.

## Methodology / Approach

**1. Document Loading**
Documents are loaded as LangChain `Document` objects, each tagged with its source filename as metadata for later citation.

**2. Chunking**
Documents are split into overlapping chunks using `RecursiveCharacterTextSplitter` (chunk size 300, overlap 50) so retrieval can return focused, relevant passages rather than entire documents.

**3. Embeddings**
Each chunk is converted into a vector embedding using `fastembed` (`BAAI/bge-small-en-v1.5`), a lightweight ONNX-based embedding model. This was chosen deliberately over `sentence-transformers`/`torch`, which repeatedly caused numpy binary-compatibility errors in the Colab environment.

**4. Vector Store & Retrieval**
Embeddings are indexed in a **FAISS** vector store for fast similarity search. At query time, the retriever returns the top-3 most relevant chunks (`k=3`) for the user's question.

**5. Generation**
Retrieved chunks are passed as context to **Llama 3 (via the Groq API)** through LangChain's `ConversationalRetrievalChain`, which generates the final answer grounded in that context.

**6. Conversational Memory**
`ConversationBufferMemory` stores the full chat history and injects it into the prompt on every turn, so follow-up questions (e.g., "Why is it useful?" after "What is RAG?") are correctly resolved without the user having to repeat themselves.

**7. Deployment**
The entire pipeline is wrapped in a **Streamlit** chat interface with a custom purple/blue gradient theme, a sidebar for the Groq API key and custom file uploads, and source citations shown under each answer. It runs directly inside the same Colab notebook via `localtunnel`, requiring no separate file or local setup.

## Model Development & Training

No model was trained from scratch for this task — the system uses:
- A pretrained embedding model (`BAAI/bge-small-en-v1.5`) for vectorizing text
- A pretrained LLM (`llama-3.1-8b-instant` via Groq) for answer generation

The engineering effort was in building the retrieval pipeline, chaining these components together correctly, and ensuring accurate context injection and memory handling — which is the core skill this task is designed to teach.

## Evaluation

The chatbot was tested with a multi-turn conversation covering direct questions and context-dependent follow-ups:

| Question | Behavior Verified |
|---|---|
| "What is RAG?" | Correct, grounded definition retrieved from `rag_systems.txt` |
| "Why is it useful?" | Follow-up resolved correctly without repeating "RAG" — confirms memory works |
| "How many internship tasks are there and what is the deadline behavior?" | Correct fact retrieved from `internship_tasks.txt` |
| "What memory class does LangChain provide and what does it do?" | Correct, grounded answer from `conversational_memory.txt` |

Each answer was returned with its source document(s) listed, confirming the response was grounded in the retrieved context rather than the model's general knowledge.

## Key Results / Observations

- **Retrieval quality**: The retriever consistently pulled the correct chunk(s) for each question, visible via the printed/displayed sources.
- **Context-awareness**: Follow-up questions referencing prior turns ("it", "that") were resolved correctly, confirming `ConversationBufferMemory` correctly carries chat history across turns.
- **Grounding**: Answers stayed grounded in the custom corpus rather than the LLM's general knowledge, which is the core value proposition of RAG — it reduces hallucination for domain-specific or private information.
- **Limitations observed**: Very short or ambiguous follow-up questions can occasionally retrieve a less-relevant chunk if the topic shifts abruptly between turns. `ConversationalRetrievalChain` mitigates this internally by condensing the question using chat history before retrieval, but it is not foolproof for highly ambiguous phrasing.
- **Engineering challenges resolved**: Several environment-level issues were debugged and fixed during development — numpy/torch ABI incompatibility (solved by replacing `sentence-transformers` with the lighter `fastembed`), a numpy/`faiss-cpu` version mismatch (solved by pinning `numpy==1.26.4`), and a LangChain memory configuration error with multiple output keys (solved by explicitly setting `output_key="answer"`).

## Tech Stack

| Component | Tool |
|---|---|
| Orchestration | LangChain |
| Embeddings | fastembed (`BAAI/bge-small-en-v1.5`) |
| Vector Store | FAISS |
| LLM | Llama 3 via Groq API |
| Memory | LangChain `ConversationBufferMemory` |
| Deployment | Streamlit (run inside Colab via localtunnel) |

## Files in This Repository

| File | Description |
|---|---|
| `Task4_RAG_Chatbot.ipynb` | Full notebook — problem statement, pipeline, testing, evaluation, and in-notebook Streamlit deployment |
| `README.md` | This file |

## Codebase

Key code snippets from the implementation:

### Chunking

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = text_splitter.split_documents(documents)
```

### Embeddings (custom fastembed wrapper)

```python
class SimpleFastEmbed(Embeddings):
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts):
        return [np.asarray(e, dtype=np.float32).tolist() for e in self._model.embed(texts)]

    def embed_query(self, text):
        return np.asarray(list(self._model.embed([text]))[0], dtype=np.float32).tolist()
```

### Vector Store & Retriever

```python
embedding_model = SimpleFastEmbed()
vectorstore = FAISS.from_documents(chunks, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

### LLM + Memory + Conversational Chain

```python
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.2)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
)
```

### Querying the Chatbot

```python
def ask(question):
    result = qa_chain.invoke({"question": question})
    print("Q:", question)
    print("A:", result["answer"])
    print("Sources:", [doc.metadata["source"] for doc in result["source_documents"]])
    return result

ask("What is RAG?")
ask("Why is it useful?")  # follow-up, tests memory ("it" = RAG)
```

## How to Run

**Option A — Google Colab (recommended, all-in-one)**
1. Open `Task4_RAG_Chatbot.ipynb` in Google Colab
2. Run all cells from top to bottom
3. Enter a free Groq API key when prompted (get one at [console.groq.com/keys](https://console.groq.com/keys))
4. The final cells launch the Streamlit app inside the notebook and print a public `https://*.loca.lt` link to open it

**Option B — Local Streamlit**
```bash
pip install langchain langchain-community langchain-text-splitters langchain-groq fastembed faiss-cpu streamlit numpy==1.26.4
streamlit run app.py
```
(Extract the `%%writefile app.py` cell content from the notebook into a standalone `app.py` file first.)

## Skills Gained

- Conversational AI development
- Document embedding and vector search (FAISS)
- Retrieval-Augmented Generation (RAG) pipeline design
- LLM integration and deployment (Groq/Llama 3 + Streamlit)
- Debugging Python ML dependency/environment conflicts
## Colab Notebook link:
https://colab.research.google.com/drive/1hSmegKiDmGsvvPgqzTA5q270TnjLr6sJ?usp=sharing
