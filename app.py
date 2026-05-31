import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
import os
import tempfile
import time
from datetime import datetime


st.set_page_config(
    page_title="DocuSense | Neural Architecture", 
    page_icon="🌌", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Playfair+Display:ital,wght@0,500;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* 
       GLOBAL NEBULA GRADIENT BACKGROUND
        */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif !important;
        background: radial-gradient(circle at 15% 50%, rgba(20, 10, 45, 1), transparent 50%), 
                    radial-gradient(circle at 85% 30%, rgba(10, 25, 60, 1), transparent 50%),
                    #030305 !important;
        background-attachment: fixed !important;
        color: #F8FAFC !important;
    }
    
    [data-testid="stHeader"] { background-color: transparent !important; }
    
    /* 
       BOLD WHITE SIDEBAR TOGGLE BUTTON 
        */
    [data-testid="collapsedControl"] {
        color: #FFFFFF !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 2px solid rgba(255, 255, 255, 0.4) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(15px);
        transition: all 0.3s ease;
        padding: 0.5rem !important;
        z-index: 999999;
    }
    [data-testid="collapsedControl"] svg {
        width: 32px !important;
        height: 32px !important;
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
        font-weight: bold !important;
    }
    [data-testid="collapsedControl"]:hover {
        background-color: rgba(168, 85, 247, 0.3) !important;
        border-color: #A855F7 !important;
        transform: scale(1.08);
        box-shadow: 0 0 20px rgba(168, 85, 247, 0.6);
    }

    /* 
       PREMIUM FLOATING CHAT INPUT
        */
    /* Wrapper positioning to fix the "floating to top" bug */
    [data-testid="stBottomBlock"] {
        background: transparent !important;
        padding-bottom: 60px !important; /* Lift above the fixed footer */
    }
    
    /* The Chat Box itself */
    [data-testid="stChatInput"] {
        background: rgba(15, 20, 35, 0.7) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(168, 85, 247, 0.5) !important;
        border-radius: 20px !important;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5), 0 0 15px rgba(168, 85, 247, 0.2) !important;
        padding: 8px 12px !important;
    }
    
    /* Chat Input Text Area */
    [data-testid="stChatInput"] textarea {
        color: #FFFFFF !important;
        background-color: transparent !important;
        font-size: 1.05rem !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #64748B !important;
    }

    /* Chat Send Button */
    [data-testid="stChatInputSubmitButton"] {
        background: rgba(168, 85, 247, 0.2) !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
        height: 100% !important;
    }
    [data-testid="stChatInputSubmitButton"]:hover {
        background: #A855F7 !important;
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(168, 85, 247, 0.5) !important;
    }
    [data-testid="stChatInputSubmitButton"] svg {
        fill: #FFFFFF !important;
    }

    /* 
       ENHANCED GLASSMORPHISM SIDEBAR
        */
    [data-testid="stSidebar"] {
        background: rgba(8, 8, 12, 0.55) !important;
        backdrop-filter: blur(25px) !important;
        -webkit-backdrop-filter: blur(25px) !important;
        border-right: 1px solid rgba(139, 92, 246, 0.15) !important;
        box-shadow: 5px 0 30px rgba(0, 0, 0, 0.5);
    }
    
    [data-testid="stSidebar"] input {
        background-color: rgba(255, 255, 255, 0.03) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        border-radius: 8px !important;
        transition: all 0.3s ease;
    }
    [data-testid="stSidebar"] input:focus {
        border-color: #A855F7 !important;
        box-shadow: 0 0 10px rgba(168, 85, 247, 0.3) !important;
    }
    
    [data-testid="stFileUploadDropzone"] {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px dashed rgba(99, 102, 241, 0.4) !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        background: rgba(99, 102, 241, 0.08) !important;
        border-color: #818CF8 !important;
    }

    [data-testid="stExpander"] {
        background-color: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px !important;
    }

    /* -----------------------------------
       TYPOGRAPHY & UI ELEMENTS
       ----------------------------------- */
    .brand-title {
        font-family: 'Playfair Display', serif;
        font-size: 4rem;
        font-weight: 600;
        letter-spacing: -1px;
        background: linear-gradient(to right, #FFFFFF, #A855F7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .brand-italic { font-style: italic; font-weight: 500; }
    .technical-label { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #818CF8; letter-spacing: 1.5px; text-transform: uppercase; }
    .technical-value { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #E2E8F0; }

    /* Primary Start Button */
    .stButton > button:first-child {
        background: linear-gradient(135deg, #4F46E5 0%, #9333EA 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        width: 100%;
        box-shadow: 0 4px 15px rgba(147, 51, 234, 0.3) !important;
    }
    .stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(147, 51, 234, 0.6) !important;
        background: linear-gradient(135deg, #6366F1 0%, #A855F7 100%) !important;
    }

    /* Suggestion Pills */
    [data-testid="stHorizontalBlock"] .stButton > button {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #CBD5E1 !important;
        border-radius: 20px !important;
        font-size: 0.85rem !important;
        padding: 6px 16px !important;
        transition: all 0.2s ease;
    }
    [data-testid="stHorizontalBlock"] .stButton > button:hover {
        background: rgba(168, 85, 247, 0.1) !important;
        border-color: #A855F7 !important;
        color: #FFFFFF !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 30px; border-bottom: 1px solid rgba(255,255,255,0.05); }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Playfair Display', serif;
        font-size: 1.15rem;
        color: #64748B;
        background-color: transparent;
        border: none;
        padding-bottom: 12px;
    }
    .stTabs [aria-selected="true"] {
        color: #F8FAFC !important;
        border-bottom: 2px solid #A855F7 !important;
    }

    /* -----------------------------------
       CHAT MESSAGES & AVATAR REFINEMENTS
       ----------------------------------- */
    .stChatMessage { 
        background: transparent !important; 
        margin-bottom: 25px !important; 
        gap: 15px !important; 
        padding: 0 10px !important;
    }
    
    /* Clean up avatar backgrounds */
    [data-testid="stChatMessageAvatarUser"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }
    [data-testid="stChatMessageAvatarAssistant"] {
        background-color: rgba(168, 85, 247, 0.15) !important;
        border: 1px solid rgba(168, 85, 247, 0.4) !important;
        border-radius: 8px !important;
    }

    /* Chat Bubbles */
    [data-testid="stChatMessageUser"] .stMarkdown {
        background: rgba(147, 51, 234, 0.15) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(147, 51, 234, 0.3);
        border-radius: 14px 14px 4px 14px;
        padding: 14px 22px !important;
        margin-left: auto;
        max-width: 80%;
    }
    [data-testid="stChatMessageAssistant"] .stMarkdown {
        background: rgba(15, 23, 42, 0.6) !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 14px 14px 14px 4px;
        border-left: 3px solid #818CF8;
        padding: 15px 22px !important;
        max-width: 90%;
        font-size: 1.02rem;
        line-height: 1.7;
        backdrop-filter: blur(10px);
    }

    /* Premium Dashboard Cards */
    .premium-panel {
        background: rgba(15, 20, 35, 0.6);
        border: 1px solid rgba(139, 92, 246, 0.15);
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 25px;
        backdrop-filter: blur(15px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    .premium-panel > div:last-child {
        border-left: 1px solid rgba(255, 255, 255, 0.05);
        padding-left: 20px;
    }
    
    /* Floating Footer */
    .premium-footer {
        position: fixed;
        bottom: 0; left: 0; width: 100%;
        background: rgba(5, 5, 8, 0.85);
        backdrop-filter: blur(20px);
        border-top: 1px solid rgba(139, 92, 246, 0.2);
        padding: 12px 0;
        text-align: center;
        z-index: 999;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 40px;
    }
    .footer-metric { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #94A3B8; display: flex; align-items: center; gap: 6px; }
    .footer-glow { color: #A855F7; font-weight: 500; }

    footer {visibility: hidden;}
    .block-container { padding-bottom: 120px !important; max-width: 1050px; }
    </style>
""", unsafe_allow_html=True)


if "messages" not in st.session_state: st.session_state.messages = []
if "vector_store" not in st.session_state: st.session_state.vector_store = None
if "file_name" not in st.session_state: st.session_state.file_name = None
if "engine_started" not in st.session_state: st.session_state.engine_started = False
if "chunk_count" not in st.session_state: st.session_state.chunk_count = 0
if "init_time" not in st.session_state: st.session_state.init_time = ""

search_tool = DuckDuckGoSearchRun()


with st.sidebar:
    st.markdown("<h2 style='font-family: \"Playfair Display\", serif; font-size: 2.2rem; font-weight: 600; text-align: center; margin-bottom:0; color: #FFFFFF;'>Docu<span style='color: #A855F7; font-style: italic;'>Sense</span></h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 0.72rem; margin-top: -2px; margin-bottom: 35px; letter-spacing: 3px; font-weight:500; color: #64748B;'>CORE SYSTEM CONTROL</p>", unsafe_allow_html=True)
    
    st.markdown("<div class='technical-label' style='margin-bottom: 8px;'>01 / GATEWAY TOKEN</div>", unsafe_allow_html=True)
    
    # Instruction text for the API Key
    st.markdown("<p style='font-size: 0.85rem; color: #CBD5E1; margin-bottom: 10px;'>Please paste your active Groq API Gateway key below to initialize the neural engine.</p>", unsafe_allow_html=True)
    
    api_key = st.text_input("Groq API Key", type="password", placeholder="Paste gsk_ token here...", label_visibility="collapsed")
    
    with st.expander("🔑 Obtain Authorization Token", expanded=False):
        st.markdown("""
        <div style='font-size: 0.85rem; color: #E2E8F0; line-height: 1.6;'>
        <b>Provisioning your secure token:</b><br><br>
        1. Access the <a href='https://console.groq.com/' target='_blank' style='color: #A855F7; text-decoration: none; font-weight: 600;'>Groq Cloud Console</a>.<br>
        2. Authenticate using your credentials.<br>
        3. Navigate to <b>API Keys</b> and execute <i>'Create API Key'</i>.<br>
        4. Paste the generated cipher into the gateway above.<br><br>
        <span style='color: #94A3B8; font-style: italic;'>* Tokens are transient and securely purged upon session termination.</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><div class='technical-label' style='margin-bottom: 8px;'>02 / CONTEXT STREAM</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.session_state.vector_store is None:
        if st.button("Initialize Pipeline", use_container_width=True):
            if not api_key or not uploaded_file:
                st.error("🔒 Token and valid Context File are mandatory.")
            else:
                st.session_state.engine_started = True
    else:
        if st.button("Purge Engine Session", use_container_width=True):
            st.session_state.vector_store = None
            st.session_state.file_name = None
            st.session_state.messages = []
            st.session_state.engine_started = False
            st.rerun()


if st.session_state.engine_started and st.session_state.vector_store is None:
    with st.status("🚀 Initiating Neural Pipeline...", expanded=True) as status:
        try:
            st.write("⏳ Provisioning secure temporary workspace...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            st.write("✅ Workspace provisioned.")

            st.write("⏳ Parsing document syntax...")
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            st.write("✅ Syntax parsed successfully.")
            
            st.write("⏳ Fragmenting text into vector chunks...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)
            st.session_state.chunk_count = len(chunks)
            st.write(f"✅ Fragmented into {st.session_state.chunk_count} discrete nodes.")

            st.write("⏳ Generating semantic embeddings...")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
            st.write("✅ Vector Matrix assembled.")
            
            st.session_state.file_name = uploaded_file.name
            st.session_state.init_time = datetime.now().strftime("%H:%M UTC")
            
            os.remove(tmp_file_path)
            status.update(label="✅ Pipeline Calibration Complete", state="complete", expanded=False)
            time.sleep(0.5)
            st.rerun()
        except Exception as e:
            status.update(label="❌ Pipeline Assembly Fault", state="error")
            st.error(f"Diagnostic Report: {e}")
            st.session_state.engine_started = False


if not st.session_state.engine_started and st.session_state.vector_store is None:
    st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 60vh; text-align: center;">
            <div class="brand-title">Docu<span class="brand-italic" style="color: #A855F7;">Sense</span></div>
            <p style="font-size: 1.1rem; color: #94A3B8; max-width: 550px; line-height: 1.8; font-weight: 400;">
                The corporate standard for complex semantic parsing and open-source intelligence. <br>Setup authentication in the sidebar to begin.
            </p>
        </div>
    """, unsafe_allow_html=True)
else:
    tab1, tab2 = st.tabs(["📄 Context Architecture", "🌐 External Intelligence"])

   
    with tab1:
        st.markdown(f"""
            <div class="premium-panel" style="display: flex; justify-content: space-between; align-items: center; padding: 16px 24px;">
                <div>
                    <span class="technical-label">ACTIVE SECTOR</span><br>
                    <span style="font-weight: 600; font-size: 1.1rem; color: #F8FAFC;">{uploaded_file.name}</span>
                </div>
                <div style="text-align: right; display: flex; gap: 40px;">
                    <div><span class="technical-label">NODES</span><br><span class="technical-value">{st.session_state.chunk_count}</span></div>
                    <div><span class="technical-label">PIPELINE</span><br><span class="technical-value">Llama 3.1</span></div>
                    <div><span class="technical-label">UPTIME</span><br><span class="technical-value">{st.session_state.init_time}</span></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if len(st.session_state.messages) == 0:
            st.markdown("<p class='technical-label' style='margin-bottom: 12px;'>SUGGESTED INITIAL ANALYTICS</p>", unsafe_allow_html=True)
            q_col1, q_col2, q_col3 = st.columns(3)
            with q_col1:
                if st.button("Generate Summary Profile"): st.session_state.next_query = "Provide a comprehensive executive summary of this document."
            with q_col2:
                if st.button("Extract Compliance Markers"): st.session_state.next_query = "What are the strict compliance rules or penalties mentioned?"
            with q_col3:
                if st.button("Isolate Target Deadlines"): st.session_state.next_query = "List all explicit deadlines and timelines found in the text."
            st.markdown("<br>", unsafe_allow_html=True)

        for i, msg in enumerate(st.session_state.messages):
            avatar_icon = "🌌" if msg["role"] == "assistant" else "💠"
            with st.chat_message(msg["role"], avatar=avatar_icon):
                st.markdown(msg["content"])
                if "citations" in msg and msg["citations"]:
                    with st.expander("Review Fragment Provenance"):
                        for cit in msg["citations"]:
                            st.markdown(f"<span class='technical-value' style='color: #A855F7; font-weight:600;'>[PAGE {cit['page']}]</span>", unsafe_allow_html=True)
                            st.markdown(f"<span style='font-family: \"Playfair Display\", serif; font-style: italic; color: #94A3B8;'>\"{cit['text']}\"</span>", unsafe_allow_html=True)

                if msg["role"] == "assistant" and i == len(st.session_state.messages) - 1:
                    st.markdown("<div style='margin-top: 20px; margin-bottom: 8px;' class='technical-label'>CONTEXTUAL DEEP DIVES:</div>", unsafe_allow_html=True)
                    s_col1, s_col2 = st.columns([1,1])
                    with s_col1:
                        if st.button("Isolate exceptions or edge cases"): 
                            st.session_state.next_query = f"What are the explicit exceptions, caveats, or unique edge cases mentioned in relation to: '{st.session_state.messages[i-1]['content']}'?"
                            st.rerun()
                    with s_col2:
                        if st.button("Corroborate structural penalties"): 
                            st.session_state.next_query = f"Are there any systemic penalties, review frameworks, or governance parameters connected to: '{st.session_state.messages[i-1]['content']}'?"
                            st.rerun()

        user_input = st.chat_input("Inquire across document arrays...")
        
        if "next_query" in st.session_state and st.session_state.next_query:
            user_input = st.session_state.next_query
            st.session_state.next_query = None

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user", avatar="💠"): 
                st.markdown(user_input)

            with st.chat_message("assistant", avatar="🌌"):
                with st.spinner("Executing Vector Search..."):
                    try:
                        llm = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key, temperature=0)
                        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                        
                        source_documents = retriever.invoke(user_input)
                        context_text = "\n\n".join([doc.page_content for doc in source_documents])
                        
                        prompt_template = ChatPromptTemplate.from_template("You are an elite corporate legal analyst. Answer using ONLY the provided context.\n\nContext:\n{context}\n\nInquiry: {question}")
                        prompt = prompt_template.format_messages(context=context_text, question=user_input)
                        response = llm.invoke(prompt)
                        
                        citations_data = [{"page": doc.metadata.get("page", 0) + 1, "text": doc.page_content[:300].strip() + "..."} for doc in source_documents]
                        
                        st.markdown(response.content)
                        with st.expander("Review Fragment Provenance"):
                            for cit in citations_data:
                                st.markdown(f"<span class='technical-value' style='color: #A855F7; font-weight:600;'>[PAGE {cit['page']}]</span>", unsafe_allow_html=True)
                                st.markdown(f"<span style='font-family: \"Playfair Display\", serif; font-style: italic; color: #94A3B8;'>\"{cit['text']}\"</span>", unsafe_allow_html=True)
                                
                        st.session_state.messages.append({"role": "assistant", "content": response.content, "citations": citations_data})
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Inference Exception: {ex}")

   
    with tab2:
        st.markdown("""
            <div style="padding-top: 10px; margin-bottom: 25px;">
                <h3 style="font-family: 'Playfair Display', serif; font-weight: 600; color: #FFFFFF; font-size:1.6rem; margin-bottom:4px;">Global Intelligence Briefing</h3>
                <p style="color: #94A3B8; font-size: 0.95rem;">Deconstruct active corporate standings via dynamic, live, open-source web aggregation.</p>
            </div>
        """, unsafe_allow_html=True)
        
        target_company = st.text_input("Target Entity Standard Nomenclature", placeholder="e.g., Stripe, Vercel, Lockheed Martin...")
        
        if st.button("Execute Intelligence Synthesis"):
            if not target_company:
                st.warning("Target configuration node parameter is empty.")
            else:
                with st.spinner(f"Initiating open-source stream parsing for '{target_company}'..."):
                    try:
                        raw_web_data = search_tool.invoke(f"Comprehensive analysis of {target_company}: business profile, operational sectors, recent executive changes, market status 2026.")
                        
                        llm = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key, temperature=0.2)
                        report_prompt = ChatPromptTemplate.from_template(
                            "You are a master corporate intelligence analyst. I have gathered live network vectors for the company: {company}. "
                            "Synthesize these unformatted nodes into a high-end Executive Briefing markdown file. "
                            "Organize it strictly with headers: Executive Core, Asset Operations, Strategic Trajectory. "
                            "Maintain a sophisticated, objective tone.\n\nRAW INTELLIGENCE INPUT:\n{web_data}"
                        )
                        
                        prompt = report_prompt.format_messages(company=target_company, web_data=raw_web_data)
                        formatted_report = llm.invoke(prompt)
                        
                        st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 30px 0;'>", unsafe_allow_html=True)
                        
                        st.markdown("<div class='premium-panel'>", unsafe_allow_html=True)
                        st.markdown(formatted_report.content)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        st.success("Synthesis Sequence Terminated Normal.")
                        
                    except Exception as e:
                        st.error(f"Intelligence Gathering Pipeline Interrupted: {e}")


st.markdown("""
    <div class="premium-footer">
        <div class="footer-metric">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#A855F7" stroke-width="2.5"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
            NODE: <span class="footer-glow">SECURE_STREAM_ONLINE</span>
        </div>
        <div class="footer-metric">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#A855F7" stroke-width="2.5"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path></svg>
            CORE: <span class="footer-glow">NEURAL_DECOUPLED_RAG</span>
        </div>
        <div class="footer-metric" style="border-left: 1px solid rgba(255,255,255,0.2); padding-left: 40px;">
            DESIGN SPECIFICATION: <span class="footer-glow" style="letter-spacing: 0.5px;">ALHAJ BAIG ARCHITECTS</span>
        </div>
    </div>
""", unsafe_allow_html=True)