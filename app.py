from vars import DEEPSEEK_API, GEMINI_API_KEY, SERPER_API_KEY, CHROMA_DB_PATH, CHROMA_COLLECTION_NAME
from funcs import *
import streamlit as st
import os
import logging
import time
from dotenv import load_dotenv
import functools
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
import chromadb
from transformers import pipeline
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Model and Component Setup ---

# Check for API Keys - Initialize these before Streamlit elements that might depend on them
try:
    if not GEMINI_API_KEY:
        logging.error("GEMINI_API_KEY not set.")
        # Display error early if possible, but might be too soon for st.error
        # We'll check again within the Streamlit part of the app.
    if not SERPER_API_KEY:
        logging.warning("SERPER_API_KEY not set. Web search will be disabled.")

    # LLM Initialization
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=GEMINI_API_KEY)

    # Embedding Function Initialization
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY,
        task_type="RETRIEVAL_QUERY"
    )

    # Classifier Initialization
    if torch.cuda.is_available():
        device_index = 0
        logging.info("GPU detected. Using GPU device: 0 for classifier.")
    else:
        device_index = -1
        logging.info("No GPU detected. Using CPU for classifier.")

    classifier = pipeline(
        task="text-classification",
        model="TBM99/Router-RAG-v2",
        tokenizer="answerdotai/ModernBERT-base",
        device=device_index,
    )
    _ = classifier("test query") # Test run
    logging.info("Classifier loaded successfully.")
    classifier_available = True

except Exception as e:
    logging.error(f"Initialization Error: {e}", exc_info=True)
    # We'll handle showing errors in the Streamlit UI part
    llm = None
    embeddings = None
    classifier = None
    classifier_available = False


# RAG Setup - Defer error handling to Streamlit UI part
vectorstore = None
try:
    if os.path.exists(CHROMA_DB_PATH):
         persistent_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
         collection_names = [col.name for col in persistent_client.list_collections()]

         if CHROMA_COLLECTION_NAME in collection_names and embeddings:
            vectorstore = Chroma(
                client=persistent_client,
                collection_name=CHROMA_COLLECTION_NAME,
                embedding_function=embeddings,
            )
            logging.info("ChromaDB vectorstore initialized.")
         else:
             logging.warning(f"Chroma collection '{CHROMA_COLLECTION_NAME}' not found or embeddings failed.")
             # RAG will not be available
    else:
        logging.warning(f"ChromaDB path '{CHROMA_DB_PATH}' not found. RAG disabled.")

except Exception as e:
    logging.error(f"ChromaDB Initialization failed: {e}", exc_info=True)
    # Error will be shown in UI

# --- LangGraph Workflow Definition ---
app_graph = None
if llm and vectorstore and classifier: # Only build graph if core components loaded
    try:
        # Use functools.partial to pass dependencies to nodes
        route_question_node_partial = functools.partial(route_question_node, classifier=classifier)
        perform_rag_node_partial = functools.partial(perform_rag_node, vectorstore=vectorstore, llm=llm)
        call_web_search_node_partial = functools.partial(call_web_search_node, SERPER_API_KEY = SERPER_API_KEY, llm = llm)
        generate_direct_llm_node_partial = functools.partial(generate_direct_llm_node ,llm = llm)
        workflow = StateGraph(GraphState)
        workflow.add_node("route_question", route_question_node_partial)
        workflow.add_node("call_web_search_node", call_web_search_node_partial)
        workflow.add_node("perform_rag_node", perform_rag_node_partial)
        workflow.add_node("generate_direct_llm_node", generate_direct_llm_node_partial)

        workflow.set_entry_point("route_question")
        workflow.add_conditional_edges(
            "route_question",
            decide_next_node,
            {
                "call_web_search_node": "call_web_search_node",
                "perform_rag_node": "perform_rag_node",
                "generate_direct_llm_node": "generate_direct_llm_node",
            },
        )
        workflow.add_edge("call_web_search_node", END)
        workflow.add_edge("perform_rag_node", END)
        workflow.add_edge("generate_direct_llm_node", END)

        app_graph = workflow.compile()
        logging.info("LangGraph workflow compiled successfully.")

    except Exception as e:
        logging.error(f"LangGraph compilation failed: {e}", exc_info=True)
        # Error handled in UI

elif llm and not classifier_available: # Handle classifier loading failure - default routing
     try:
         logging.warning("Classifier failed to load. Building graph with LLM-only routing.")
         def fallback_router_node(state: GraphState) -> dict:
            logging.warning("Classifier not available, defaulting route to LLM.")
            return {"datasource": "llm"}

         # Only define nodes that can run without the classifier/retriever if needed
         generate_direct_llm_node_partial = functools.partial(generate_direct_llm_node, llm=llm)

         workflow = StateGraph(GraphState)
         workflow.add_node("route_question", fallback_router_node) # Use fallback
         workflow.add_node("generate_direct_llm_node", generate_direct_llm_node_partial)

         workflow.set_entry_point("route_question")
         # Simplified conditional edge directly to LLM node
         workflow.add_conditional_edges(
            "route_question",
            lambda state: "generate_direct_llm_node", # Always go to LLM
            {"generate_direct_llm_node": "generate_direct_llm_node"}
         )
         workflow.add_edge("generate_direct_llm_node", END)

         app_graph = workflow.compile()
         logging.info("LangGraph workflow compiled (LLM-only fallback).")

     except Exception as e:
        logging.error(f"LangGraph compilation failed (LLM-only fallback): {e}", exc_info=True)


# --- Backend Function ---
def run_query(question: str) -> dict:
    """Invokes the compiled LangGraph app and returns the final state."""
    if not app_graph:
        return {"error": "Graph not compiled due to initialization errors.", "question": question, "generation": "Error: Application backend not ready."}

    inputs = {"question": question}
    try:
        # Using invoke for simplicity
        final_state = app_graph.invoke(inputs, {"recursion_limit": 10})
        # Ensure final_state is a dict, handle potential None or other types
        if not isinstance(final_state, dict):
             logging.error(f"Graph invocation returned non-dict type: {type(final_state)}")
             return {"error": "Invalid response type from graph.", "question": question, "generation": "Error: Backend returned invalid data."}
        return final_state # Return the whole state dict
    except Exception as e:
        logging.error(f"Error during graph invocation for query '{question}': {e}", exc_info=True)
        return {"error": str(e), "question": question, "generation": f"An error occurred processing your request: {e}"}


# --- 2. Streamlit App Interface ---

st.set_page_config(page_title="Crypto Bot", layout="wide")

# --- App Styling ---
st.markdown("""
<style>
    /* Reduce default Streamlit padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    /* Style chat messages */
    .stChatMessage {
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #eee; /* Add subtle border */
    }
    /* User message specific style */
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p { /* Target paragraphs inside user message */
         /* Add specific user styling if needed */
         /* color: #333; */
    }
    /* Assistant message specific style */
     div[data-testid="stChatMessage"]:has(span[title="assistant"]) { /* Target assistant messages */
        background-color: #f8f9fa; /* Light background for assistant */
     }

    /* Sidebar styling */
     [data-testid="stSidebar"] {
         /* background-color: #f8f9fa; */ /* Let theme handle sidebar background */
         padding-top: 1rem;
     }
     [data-testid="stSidebar"] .stButton button {
         width: 100%;
         justify-content: flex-start;
         padding-left: 1rem;
         margin-bottom: 0.5rem;
         border-radius: 8px;
         border: 1px solid #d1d5db; /* Add border to button */
         background-color: #ffffff;
     }
      [data-testid="stSidebar"] .stButton button:hover {
         background-color: #f3f4f6; /* Slightly darker hover */
         border-color: #adb5bd;
      }
       [data-testid="stSidebar"] .stButton button:active {
          background-color: #e5e7eb; /* Darker active state */
       }

    /* Center the initial placeholder */
     .empty-chat-placeholder {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 60vh; /* Adjust height as needed */
        text-align: center;
        color: #6c757d; /* Slightly darker grey */
        font-size: 1.1rem;
     }
      .empty-chat-placeholder h1 {
         font-size: 1.8rem;
         font-weight: 600;
         margin-bottom: 0.5rem;
         color: #495057;
      }
</style>
""", unsafe_allow_html=True)

# --- Error Checks ---
if not GEMINI_API_KEY:
    st.error("🔴 FATAL: GEMINI_API_KEY environment variable not set! The application cannot function.", icon="🚨")
    st.stop()
if not SERPER_API_KEY:
    st.warning("🟡 SERPER_API_KEY not set. Web search will be disabled.", icon="⚠️")
if not llm:
     st.error("🔴 LLM failed to initialize. Cannot proceed.", icon="🚨")
     st.stop()
if not embeddings:
     st.error("🔴 Embedding model failed to initialize. RAG might be affected.", icon="🚨")
     # Allow to proceed but warn RAG is likely broken
if not vectorstore:
     st.warning("🟡 ChromaDB failed to initialize. RAG functionality is disabled.", icon="⚠️")
if not classifier_available:
     st.warning("🟡 Classifier failed to load. Routing will default to LLM.", icon="⚠️")
if not app_graph:
     st.error("🔴 Backend graph failed to compile. Application cannot process queries.", icon="🚨")
     st.stop()


# --- Sidebar ---
with st.sidebar:
    st.title("🤖 Crypto Bot")
    st.markdown("---")

    # New Chat button
    if st.button("➕ New Chat", key="new_chat_button"):
        st.session_state.messages = [] # Clear history
        st.session_state.run_details = [] # Clear details
        logging.info("New chat started.")
        st.rerun() # Rerun the app to reflect the cleared state

    st.markdown("---")
    st.subheader("Run Details")
    # Placeholder for displaying details of the last run
    if "run_details" not in st.session_state:
        st.session_state.run_details = []

    if st.session_state.run_details:
         # Display details of the *last* run
         last_run = st.session_state.run_details[-1]
         st.write(f"**Last Question:**")
         st.caption(f"{last_run.get('question', 'N/A')}")
         st.write(f"**Datasource Used:**")
         st.caption(f"`{last_run.get('datasource', 'N/A')}`")
         if 'response_time' in last_run:
             st.write(f"**Response Time:**")
             st.caption(f"{last_run['response_time']}")
         if 'error' in last_run:
              st.write("**Status:**")
              st.caption(f"Error: {last_run['error'][:100]}...") # Show truncated error
    else:
        st.caption("Ask a question to see run details.")

    st.markdown("---")
    st.caption("Powered by LangGraph & Gemini")


# --- Main Chat Area ---

# Initialize chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display initial placeholder if chat history is empty
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-chat-placeholder">
        <h1>How can I help you today?</h1>
        <p>Ask me about crypto, DeFi, market news, or general topics.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"]) # Use markdown for rich formatting

# Accept user input using st.chat_input
if prompt := st.chat_input("Ask me anything about Crypto..."):
    # Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        start_time = time.time()

        # Call the backend function
        response_data = run_query(prompt) # Expects a dictionary
        end_time = time.time()

        # Extract information from the response dictionary
        if isinstance(response_data, dict):
            full_response = response_data.get('generation', "Sorry, I couldn't generate a response.")
            datasource = response_data.get('datasource', 'N/A') # Get datasource from response
            error_info = response_data.get('error')
        else:
            # Handle unexpected response format
            logging.error(f"Unexpected response type from run_query: {type(response_data)}")
            full_response = "Error: Received an unexpected response format from the backend."
            datasource = "Error"
            error_info = "Invalid response format"


        # Prepare run details
        run_info = {
            "question": prompt,
            "datasource": datasource,
            "response_time": f"{end_time - start_time:.2f}s"
        }
        if error_info:
            run_info["error"] = str(error_info)
            # Ensure the displayed response indicates an error if one occurred backend
            if not full_response.startswith("Sorry, an error occurred:") and not full_response.startswith("An error occurred"):
                 full_response = f"Sorry, an error occurred: {error_info}"


        # Update the placeholder with the final response
        message_placeholder.markdown(full_response)

        # Store run details for the sidebar
        if "run_details" not in st.session_state:
             st.session_state.run_details = []
        st.session_state.run_details.append(run_info)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Trigger a rerun to update the sidebar immediately after processing
        # This can sometimes cause a slight visual jump, but ensures sidebar is current.
        # Consider removing if the jump is too disruptive.
        st.rerun()
