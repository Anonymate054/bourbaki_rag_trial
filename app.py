import os
import sys
import time
import pandas as pd
import streamlit as st

# Setup Path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.rag_pipeline.pipeline import RAGPipelineRunner, PipelineConfig
from src.rag_pipeline.memory import MultiChatMemoryManager
from tunnel_manager import TunnelManager

# Page Setup - AUTO Responsive Sidebar State for Mobile & Desktop
st.set_page_config(
    page_title="RAG Playground & Visual Database Showcase - LFPIORPI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="auto"  # Automatically collapses on mobile screens (<768px)
)

# Custom Responsive CSS Styling for Mobile, Tablet, and Desktop Devices
st.markdown("""
<style>
    /* Global Container Responsive Adjustments */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100%;
    }
    
    /* Responsive Chat Bubbles */
    .stChatMessage[data-testid="stChatMessageUser"] {
        flex-direction: row-reverse;
        text-align: right;
        background-color: #E0F2FE;
        border-radius: 12px;
        margin-left: 10%;
    }
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: #F8FAFC;
        border-radius: 12px;
        margin-right: 10%;
    }

    /* Mobile Viewport Optimizations (<768px) */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 0.5rem;
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }
        
        .stChatMessage[data-testid="stChatMessageUser"] {
            margin-left: 2%;
        }
        .stChatMessage[data-testid="stChatMessageAssistant"] {
            margin-right: 2%;
        }
        
        /* Metric Cards Responsive Stacking */
        [data-testid="stMetricValue"] {
            font-size: 1.2rem !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.8rem !important;
        }
        
        /* Dataframe Scroll Container */
        .stDataFrame {
            width: 100% !important;
            overflow-x: auto !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session Memory & Tunnel Manager in Streamlit Session State
if "memory_mgr" not in st.session_state:
    st.session_state.memory_mgr = MultiChatMemoryManager()

tunnel_mgr = TunnelManager(8501)

# Ensure current session exists
sessions = st.session_state.memory_mgr.list_sessions()
if not sessions:
    new_s_id = st.session_state.memory_mgr.create_session()
    st.session_state.current_session_id = new_s_id
    sessions = st.session_state.memory_mgr.list_sessions()
elif "current_session_id" not in st.session_state or not st.session_state.current_session_id:
    st.session_state.current_session_id = sessions[0]["session_id"]

@st.cache_resource
def get_pipeline_runner():
    config = PipelineConfig()
    runner = RAGPipelineRunner(config)
    runner.initialize()
    return runner

runner = get_pipeline_runner()

# --- SIDEBAR: CHAT MEMORY, PLAYGROUND & INTERNET TUNNEL ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/law.png", width=64)
    st.title("🎛️ RAG Control Panel")
    
    # 1. Multi-Chat Session Management & Deletion / Locking
    st.subheader("💬 Gestión de Conversaciones")
    sessions = st.session_state.memory_mgr.list_sessions()
    if not sessions:
        st.session_state.memory_mgr.create_session()
        sessions = st.session_state.memory_mgr.list_sessions()
        
    if st.button("➕ Crear Nuevo Chat", use_container_width=True):
        new_id = st.session_state.memory_mgr.create_session()
        st.session_state.current_session_id = new_id
        st.rerun()
        
    session_titles = {s["session_id"]: f"{'🔒 ' if s.get('is_locked') else ''}{s['title']} ({s['message_count']} msgs)" for s in sessions}
    session_keys = list(session_titles.keys())
    curr_idx = session_keys.index(st.session_state.current_session_id) if st.session_state.current_session_id in session_keys else 0
    
    def on_session_select():
        st.session_state.current_session_id = st.session_state.session_selectbox_key
        
    selected_session = st.selectbox(
        "Seleccionar Chat:",
        options=session_keys,
        format_func=lambda x: session_titles.get(x, x),
        index=curr_idx,
        key="session_selectbox_key",
        on_change=on_session_select
    )
    st.session_state.current_session_id = selected_session
        
    current_session_data = st.session_state.memory_mgr.get_session(st.session_state.current_session_id)
    is_locked = current_session_data.get("is_locked", False)
    
    # Action Buttons: Lock & Delete
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        lock_label = "🔓 Desbloquear" if is_locked else "🔒 Bloquear"
        if st.button(lock_label, use_container_width=True):
            st.session_state.memory_mgr.toggle_lock(st.session_state.current_session_id)
            st.rerun()
            
    with col_btn2:
        if st.button("🗑️ Borrar", use_container_width=True, type="secondary"):
            st.session_state.memory_mgr.delete_session(st.session_state.current_session_id)
            remaining_sessions = st.session_state.memory_mgr.list_sessions()
            if remaining_sessions:
                st.session_state.current_session_id = remaining_sessions[0]["session_id"]
            else:
                st.session_state.current_session_id = st.session_state.memory_mgr.create_session()
            st.rerun()
    
    st.divider()

    # 2. PUBLIC INTERNET HTTPS TUNNEL CONTROL
    st.subheader("🌐 Acceso Remoto Internet (Túnel HTTPS)")
    t_status = tunnel_mgr.get_status()
    
    if t_status["active"]:
        st.success(f"🟢 **Túnel {t_status.get('engine', '')} Activo**")
        st.code(t_status["url"], language="text")
        st.link_button("🔗 Abrir Sitio Público en Internet", t_status["url"], use_container_width=True)
        if st.button("🛑 Apagar Túnel Público", use_container_width=True, type="primary"):
            tunnel_mgr.stop_tunnel()
            st.rerun()
    else:
        st.info("🔴 **Túnel Inactivo (Solo Acceso Local)**")
        tunnel_engine = st.radio(
            "Seleccionar Motor de Túnel:",
            options=["ngrok", "cloudflare"],
            format_func=lambda x: "⚡ NGROK (Recomendado 200 OK)" if x == "ngrok" else "🌐 Cloudflare Quick Tunnel"
        )
        
        ngrok_token = ""
        if tunnel_engine == "ngrok":
            ngrok_token = st.text_input(
                "🔑 Authtoken NGROK (Gratuito):",
                type="password",
                placeholder="Pega tu token de dashboard.ngrok.com...",
                help="Obtén tu token gratuito en 10s en: https://dashboard.ngrok.com/get-started/your-authtoken"
            )
            
        if st.button("🚀 Encender Túnel Público", use_container_width=True):
            with st.spinner("Creando túnel seguro HTTPS..."):
                try:
                    if tunnel_engine == "ngrok":
                        tunnel_mgr.start_ngrok_tunnel(authtoken=ngrok_token)
                    else:
                        tunnel_mgr.start_cloudflare_tunnel()
                except Exception as ex:
                    st.error(f"Error al iniciar túnel: {ex}")
            st.rerun()

    st.divider()
    
    # 3. EXPERIMENTAL PLAYGROUND PARAMETERS (Toggles, Sliders & LLM Mode Selection)
    st.subheader("🧪 Hiperparámetros RAG (Playground)")
    
    retrieval_mode = st.radio(
        "1. Algoritmo de Búsqueda:",
        options=["hybrid", "dense", "sparse"],
        format_func=lambda x: (
            "🔀 Híbrida (Fusión RRF: Semántica + Palabras Clave)" if x == "hybrid" else
            ("⚡ Densa (Búsqueda Semántica: Vectorial FAISS Coseno GPU)" if x == "dense" else
             "🔤 Dispersa (Búsqueda por Palabras Clave: BM25 Okapi)")
        ),
        help="Permite alternar entre búsqueda vectorial densa (semántica), dispersa (palabras clave) o fusión híbrida RRF."
    )
    
    enable_rerank = st.toggle(
        "2. Re-ranking (Cross-Encoder)",
        value=True,
        help="Activa la atención cruzada para reordenar los top candidatos pasándolos por el re-ranker."
    )
    
    enable_router = st.toggle(
        "3. Enrutador Explícito de Artículos",
        value=True,
        help="Prioriza automáticamente al 100% los artículos explícitamente nombrados en la pregunta (ej. 'Artículo 17')."
    )
    
    preprocessing_mode = st.radio(
        "4. Preprocesamiento de Texto:",
        options=["raw", "clean"],
        format_func=lambda x: "📄 Texto Raw (Natural Legal)" if x == "raw" else "🧹 Texto Limpio (Minúsculas / Normalizado)",
        help="Compara el rendimiento sobre texto natural vs. texto filtrado."
    )
    
    top_k = st.slider("5. Top-K Chunks Recuperados:", min_value=1, max_value=5, value=3)
    
    # LLM Execution Provider (Local GPU Ollama vs Cloud API)
    # INSTRUCCIONES:
    # 1. Para probar LLMs localmente en GPU: Asegúrate de tener Ollama ejecutándose en http://localhost:11434
    # 2. Para usar una API en la nube (OpenAI / Groq / OpenRouter): Selecciona "API Externa" e ingresa tu API Key o configúrala en .env
    llm_provider_choice = st.radio(
        "6. Proveedor de Modelo LLM:",
        options=["ollama", "cloud_api"],
        format_func=lambda x: "💻 Local GPU (Ollama)" if x == "ollama" else "☁️ API Externa (OpenAI / Groq)"
    )
    
    cloud_api_key = ""
    if llm_provider_choice == "cloud_api":
        cloud_api_key = st.text_input(
            "🔑 API Key de Cloud (OpenAI/Groq):",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            placeholder="sk-...",
            help="Ingresa tu API Key para generar respuestas usando modelos externos en la nube."
        )
        ollama_model = st.selectbox(
            "Modelo Cloud:",
            options=["gpt-4o-mini", "llama-3.3-70b-versatile", "gpt-3.5-turbo"],
            index=0
        )
    else:
        ollama_model = st.selectbox(
            "Modelo Local (Ollama GPU):",
            options=["qwen2.5:3b", "llama3.2:3b", "phi3.5:latest", "qwen2.5:7b"],
            index=0
        )

# --- MAIN CONTENT AREA WITH TOP-LEVEL TABS ---
tab_chat, tab_db = st.tabs(["💬 RAG Playground & Chatbot", "🗄️ Visualizador de Base de Datos Vectorial & Showcase"])

# ==========================================
# TAB 1: RAG PLAYGROUND & CHATBOT INTERFAZ
# ==========================================
with tab_chat:
    session_data = st.session_state.memory_mgr.get_session(st.session_state.current_session_id)
    is_locked = session_data.get("is_locked", False)
    messages = session_data.get("messages", [])

    title_cols = st.columns([0.8, 0.2])
    with title_cols[0]:
        st.title("🏛️ RAG Playground & Chatbot Legal (LFPIORPI)")
        st.caption("Laboratorio interactivo de inspección de Chunks, Métricas de Similitud Coseno, BM25 y Re-ranking en tiempo real.")
    with title_cols[1]:
        if is_locked:
            st.error("🔒 Chat Bloqueado")
        else:
            st.success("🟢 Chat Activo")

    st.divider()

    if not messages:
        with st.container(border=True):
            st.subheader("👋 ¡Bienvenido al RAG Playground!")
            st.write("Esta interfaz te permite experimentar en tiempo real con todos los parámetros de la arquitectura RAG. Observa cómo cambia la similitud coseno, el puntaje BM25 y el orden de los Chunks al modificar los controles en la barra lateral.")
            
            st.info("💡 **Preguntas Sugeridas para Experimentar:**\n\n"
                    "• *¿Cuál es el objeto de la Ley según el Artículo 2?*\n"
                    "• *¿Qué actividades se consideran vulnerables en el Artículo 17?*\n"
                    "• *¿Cuáles son las sanciones del Artículo 62?*\n"
                    "• *¿Por cuánto tiempo se deben conservar los documentos de clientes según el Artículo 18?*")

    def render_inspection_panel(metadata: dict):
        metrics = metadata.get("metrics", {})
        contextos = metadata.get("contextos", [])
        prompt_inyectado = metadata.get("prompt_inyectado", "")
        
        with st.expander("🔬 INSPECTOR RAG PLAYGROUND: Estadísticas, Chunks y Métricas de Decisión", expanded=True):
            st.markdown("### 📊 Métricas de Rendimiento de la Inferencia")
            cols = st.columns(5)
            cols[0].metric("Latencia Búsqueda", f"{metrics.get('retrieval_time_ms', 0)} ms")
            cols[1].metric("Tiempo Generación", f"{metrics.get('generation_time_sec', 0)} s")
            cols[2].metric("Modo Búsqueda", str(metrics.get('retrieval_mode', 'hybrid')).upper())
            cols[3].metric("Re-ranking", "ON" if metrics.get('rerank_enabled') else "OFF")
            cols[4].metric("Article Router", "ON" if metrics.get('router_enabled') else "OFF")
            
            if contextos:
                st.markdown("### 🧱 Tabla Comparativa de Chunks Escogidos y Puntajes")
                
                table_data = []
                for ctx in contextos:
                    table_data.append({
                        "Rango Final": f"Posición #{ctx.get('rank', 1)}",
                        "Artículo": ctx.get('articulo'),
                        "Chunk ID": ctx.get('chunk_id'),
                        "Similitud Coseno (FAISS)": ctx.get('faiss_cosine_sim', 0.0),
                        "Puntaje BM25": ctx.get('bm25_score', 0.0),
                        "Score Cross-Encoder": ctx.get('cross_encoder_score', 'N/A'),
                        "Router Boost": "✅ Sí" if ctx.get('is_boosted_by_router') else "No"
                    })
                df_chunks = pd.DataFrame(table_data)
                st.dataframe(df_chunks, use_container_width=True)
                
                st.markdown("### 📜 Inspección de Texto y Contenido de los Chunks")
                for idx, ctx in enumerate(contextos, 1):
                    st.markdown(f"**Chunk #{idx} — {ctx['articulo']} (ID: `{ctx['chunk_id']}`)**")
                    sub_cols = st.columns(4)
                    sub_cols[0].caption(f"**Similitud Coseno:** {ctx.get('faiss_cosine_sim')}")
                    sub_cols[1].caption(f"**BM25 Score:** {ctx.get('bm25_score')}")
                    sub_cols[2].caption(f"**Cross-Encoder:** {ctx.get('cross_encoder_score')}")
                    sub_cols[3].caption(f"**Enrutado:** {'✅ Sí' if ctx.get('is_boosted_by_router') else 'No'}")
                    st.info(ctx['texto'])
                    
            if prompt_inyectado:
                with st.expander("📄 Ver Prompt Completo Inyectado al LLM"):
                    st.code(prompt_inyectado, language="markdown")

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        metadata = msg.get("metadata", {})
        
        with st.chat_message(role):
            st.markdown(content)
            if role == "assistant" and metadata:
                render_inspection_panel(metadata)

    if is_locked:
        st.warning("🔒 Esta conversación está bloqueada. Haz clic en '🔓 Desbloquear' en la barra lateral para volver a enviar mensajes.")
    else:
        if user_input := st.chat_input("Escribe tu consulta sobre la Ley de Prevención e Identificación de Recursos Ilícitos..."):
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.memory_mgr.add_message(st.session_state.current_session_id, "user", user_input)
            
            override_config = {
                "retrieval_mode": retrieval_mode,
                "enable_rerank": enable_rerank,
                "enable_router": enable_router,
                "preprocessing_mode": preprocessing_mode,
                "top_k": top_k,
                "ollama_model": ollama_model,
                "provider": llm_provider_choice,
                "api_key": cloud_api_key
            }
            
            with st.chat_message("assistant"):
                with st.spinner("🔍 Consultando ley, calculando métricas de decisión y ejecutando inferencia..."):
                    result = runner.query(user_input, override_config=override_config)
                    
                st.markdown(result["response"])
                render_inspection_panel(result)
                        
            st.session_state.memory_mgr.add_message(st.session_state.current_session_id, "assistant", result["response"], metadata=result)
            st.rerun()

# =======================================================
# TAB 2: VISUALIZADOR DE BASE DE DATOS VECTORIAL & SHOWCASE
# =======================================================
with tab_db:
    st.header("🗄️ Explorador de Base de Datos Vectorial & Chunks (Showcase Visual)")
    st.caption("Visualiza, filtra y analiza interactivamente la estructura de datos indexada en FAISS GPU y BM25.")
    
    # 1. Encabezado de Métricas de la Base de Datos con Fallbacks Robustos
    all_chunks = runner.chunks
    for c in all_chunks:
        if "longitud_caracteres" not in c:
            c["longitud_caracteres"] = len(c.get("texto", ""))
        if "total_palabras" not in c:
            c["total_palabras"] = len(c.get("texto", "").split())
            
    df_raw = pd.DataFrame(all_chunks)
    
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Total Chunks Indexados", len(df_raw))
    kpi_cols[1].metric("Dimensión Vectorial", "384 Dims (all-MiniLM-L6-v2)")
    kpi_cols[2].metric("Aceleración GPU", "NVIDIA GTX 1070 (CUDA)")
    kpi_cols[3].metric("Total Caracteres", f"{df_raw['longitud_caracteres'].sum():,} chars")
    
    st.divider()
    
    # 2. Panel de Filtros Interactivos & Búsqueda
    st.subheader("🔍 Filtros Dinámicos de Búsqueda")
    
    f_col1, f_col2, f_col3 = st.columns([0.4, 0.3, 0.3])
    with f_col1:
        search_query = st.text_input("🔎 Buscar palabra clave en el texto:", placeholder="Ej. sanciones, UMA, fideicomiso...")
    with f_col2:
        article_options = sorted(list(df_raw["articulo"].unique()))
        selected_articles = st.multiselect("📌 Filtrar por Artículos Específicos:", options=article_options)
    with f_col3:
        min_len, max_len = int(df_raw["longitud_caracteres"].min()), int(df_raw["longitud_caracteres"].max())
        char_range = st.slider("📏 Rango de Caracteres por Chunk:", min_value=min_len, max_value=max_len, value=(min_len, max_len))
        
    s_col1, s_col2, s_col3 = st.columns([0.35, 0.35, 0.3])
    with s_col1:
        sort_column = st.selectbox(
            "🔀 Ordenar Datos Por:",
            options=["chunk_id", "articulo", "longitud_caracteres", "total_palabras"],
            format_func=lambda x: {
                "chunk_id": "Chunk ID",
                "articulo": "Artículo",
                "longitud_caracteres": "Longitud de Caracteres",
                "total_palabras": "Total de Palabras"
            }.get(x, x)
        )
    with s_col2:
        sort_order = st.radio("⬆️⬇️ Dirección:", options=["Ascendente", "Descendente"], horizontal=True)
    with s_col3:
        visible_cols = st.multiselect(
            "👁️ Columnas Visibles:",
            options=list(df_raw.columns),
            default=["chunk_id", "articulo", "tipo", "longitud_caracteres", "total_palabras", "texto"]
        )

    # Filtering Logic
    df_filtered = df_raw.copy()
    if search_query:
        df_filtered = df_filtered[df_filtered["texto"].str.contains(search_query, case=False, na=False)]
    if selected_articles:
        df_filtered = df_filtered[df_filtered["articulo"].isin(selected_articles)]
    df_filtered = df_filtered[
        (df_filtered["longitud_caracteres"] >= char_range[0]) & 
        (df_filtered["longitud_caracteres"] <= char_range[1])
    ]
    
    # Sorting Logic
    ascending = True if sort_order == "Ascendente" else False
    df_filtered = df_filtered.sort_values(by=sort_column, ascending=ascending)
    
    st.success(f"📊 Mostrando **{len(df_filtered)}** de **{len(df_raw)}** Chunks que coinciden con los criterios de búsqueda.")
    
    # 3. Dataframe Interactivo Showcase
    st.subheader("📋 Tabla Interactiva de la Base de Datos")
    st.dataframe(
        df_filtered[visible_cols],
        use_container_width=True,
        height=380
    )
    
    st.divider()
    
    # 4. Tarjetas Gráficas de Inspección (Showcase Cards)
    st.subheader("🧱 Visor Gráfico de Chunks (Showcase Cards)")
    show_cards = st.checkbox("Mostrar Chunks como Tarjetas Interactivas", value=True)
    
    if show_cards:
        for idx, row in df_filtered.iterrows():
            with st.container(border=True):
                card_title_col, card_meta_col = st.columns([0.7, 0.3])
                with card_title_col:
                    st.markdown(f"#### 📜 {row['articulo']} — `ID: {row['chunk_id']}`")
                with card_meta_col:
                    st.caption(f"📏 **{row['longitud_caracteres']} chars** | 📝 **{row['total_palabras']} palabras**")
                    
                st.info(row["texto"])
                
    st.divider()
    
    # 5. Gráficos Estadísticos del Corpus (Corpus Analytics)
    st.subheader("📈 Analítica Visual del Corpus Legal Indexado")
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("##### 📏 Distribución de Caracteres por Chunk")
        char_counts = df_raw.set_index("chunk_id")["longitud_caracteres"]
        st.bar_chart(char_counts)
        
    with chart_col2:
        st.markdown("##### 📌 Número de Chunks por Artículo")
        art_counts = df_raw["articulo"].value_counts().head(15)
        st.bar_chart(art_counts)
