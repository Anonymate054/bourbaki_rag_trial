# GRAPH.md: Topología del Grafo de Código RAG (DAG Architecture)

Este documento describe la arquitectura modular basada en un **Grafo Acíclico Dirigido (DAG)** para que cualquier nuevo agente de IA o desarrollador pueda entender, extender o reemplazar nodos de la canalización de forma *Plug & Play*.

---

## 📐 Topología del Grafo de Ejecución

```mermaid
graph TD
    Sub_Doc["Documento: Compilado_LFPIORPI20mayo2021.txt"] --> Node_1["Node 1: DocumentLoaderNode<br/>(src/rag_pipeline/nodes/loader.py)"]
    Node_1 --> Node_2["Node 2: StructuralChunkerNode<br/>(src/rag_pipeline/nodes/chunker.py)"]
    
    Node_2 --> Node_3A["Node 3A: FAISSVectorStoreNode<br/>(src/rag_pipeline/nodes/vector_store.py)"]
    Node_2 --> Node_3B["Node 3B: BM25StoreNode<br/>(src/rag_pipeline/nodes/bm25_store.py)"]
    
    Sub_User["Consulta de Usuario"] --> Node_Run["DAG Orchestrator: RAGPipelineRunner<br/>(src/rag_pipeline/pipeline.py)"]
    
    Node_Run --> Node_3A
    Node_Run --> Node_3B
    
    Node_3A & Node_3B --> Node_4["Node 4: ReciprocalRankFusionNode (RRF)<br/>(src/rag_pipeline/nodes/fusion.py)"]
    
    Node_4 --> Node_5["Node 5: CrossEncoderRerankerNode<br/>(src/rag_pipeline/nodes/reranker.py)"]
    
    Node_5 --> Node_6["Node 6: LocalLLMNode (Ollama REST API)<br/>(src/rag_pipeline/nodes/llm.py)"]
    
    Node_6 --> Sub_UI["Streamlit ChatGPT Web UI<br/>(app.py)"]
```

---

## 🛠️ Guía de Nodos para Agentes AI

| ID Nodo | Archivo Fuente | Entrada | Salida | Responsabilidad |
| :--- | :--- | :--- | :--- | :--- |
| **Node 1** | `nodes/loader.py` | Ruta de archivo `.txt` | Texto plano sin cabeceras DOF | Ingesta y filtrado de cabeceras oficiales. |
| **Node 2** | `nodes/chunker.py` | Texto plano | Lista de dicts `chunks` | Splitting estructural por artículos de ley. |
| **Node 3A** | `nodes/vector_store.py` | Chunks | Íntegro FAISS en GPU CUDA | Generación de embeddings densos y Parquet. |
| **Node 3B** | `nodes/bm25_store.py` | Chunks | Índice Okapi BM25 | Tokenización y búsqueda dispersa. |
| **Node 4** | `nodes/fusion.py` | Rankings Densos + Dispersos | Candidatos Top-N RRF | Fusión recíproca de clasificaciones ($k=60$). |
| **Node 5** | `nodes/reranker.py` | Query + Candidatos RRF | Top-K Chunks Reordenados | Cross-Attention scoring sobre GPU. |
| **Node 6** | `nodes/llm.py` | Prompt Inyectado con Fuentes | Texto de Respuesta | Conector cliente REST Ollama `localhost:11434`. |
| **Runner** | `pipeline.py` | Configuración + User Query | Dict `result` completo | Orquestador ejecutor del DAG con toggles. |
| **UI** | `app.py` | Configuración de Toggles & Chat | Renderizado Interactivo | Interfaz gráfica Web Streamlit estilo ChatGPT. |
| **Memory** | `memory.py` | `session_id`, `message` | JSON Persistente | Gestor de múltiples conversaciones. |

---

## 🔌 Cómo Agregar o Reemplazar un Nodo

Para añadir una nueva funcionalidad (ej. un nuevo filtro de moderación o un modelo de embeddings distinto):
1. Crea una clase de nodo en `src/rag_pipeline/nodes/mi_nodo.py` que exponga una función `.run()` o `.process()`.
2. Importa e instancia el nodo dentro de `RAGPipelineRunner` en `src/rag_pipeline/pipeline.py`.
3. Si el nodo requiere un control interactivo en la Web UI, añade la variable de control en el panel lateral de `app.py`.
