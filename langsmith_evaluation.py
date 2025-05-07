import os
import asyncio
import logging
from langsmith import Client
from langsmith.evaluation import evaluate, RunEvaluator, EvaluationResult
from langsmith.schemas import Example, Run
from langchain.evaluation import load_evaluator
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_deepseek import ChatDeepSeek
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
import chromadb
from transformers import pipeline
import torch
import functools
import time
from dotenv import load_dotenv # Keep dotenv import if used elsewhere, though keys are hardcoded here

# Assuming GraphState and node functions (route_question_node, etc.) are defined here
# If they are in a separate file 'funcs.py', this import is correct
from funcs import *

# --- Configuration & Initialization ---

# Using uppercase for constants is conventional
GEMINI_API_KEY = "" # WARNING: Hardcoded API Key
SERPER_API_KEY = "" # WARNING: Hardcoded API Key
LANGSMITH_API_KEY = "" # WARNING: Hardcoded API Key
DEEPSEEK_API = ""

# Use a more standard path name (avoiding spaces)
CHROMA_DB_PATH = "./doc_db" # Corrected path name convention
CHROMA_COLLECTION_NAME = "crypto_db"
LANGSMITH_PROJECT_NAME = os.getenv("LANGCHAIN_PROJECT", "Crypto Bot Evaluation") # Define project name, fallback if not set
DATASET_NAME = "Crypto Bot Eval Dataset V1" # Define a dataset name

# Configure logging (only once at the start)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Evaluation Dataset ---
# Renamed evaluation_dataset to EVALUATION_DATASET for constant convention
EVALUATION_DATASET = [
    # --- RAG Examples (Targeting Specific Keywords/Mechanisms) ---
    {"question": "Explain the role of Scaled Balance (ScB) in Aave V2 internal accounting.", "expected_datasource": "rag"},
    {"question": "How does Credit Delegation work in Aave V2 without collateral transfer?", "expected_datasource": "rag"},
    {"question": "What is the mathematical Value Function (V) used in Balancer pools?", "expected_datasource": "rag"},
    {"question": "Describe the process for Single-Asset Deposit/Withdrawal in Balancer.", "expected_datasource": "rag"},
    {"question": "What is the specific Proof-of-Work algorithm mentioned for Bitcoin (based on Hashcash)?", "expected_datasource": "rag"},
    {"question": "How does the Timestamp Server concept work in the Bitcoin whitepaper?", "expected_datasource": "rag"},
    {"question": "What are the components of a Hybrid Smart Contract according to the Chainlink 2.0 paper?", "expected_datasource": "rag"},
    {"question": "Explain the concept of Fair Sequencing Services (FSS) as proposed by Chainlink.", "expected_datasource": "rag"},
    {"question": "What is the function of the 'Comet' contract in Compound III?", "expected_datasource": "rag"},
    {"question": "How does the 'Absorb' function handle liquidations in Compound V3?", "expected_datasource": "rag"},
    {"question": "What is the purpose of the Inter-blockchain Communication (IBC) protocol in Cosmos?", "expected_datasource": "rag"},
    {"question": "Explain the 'Unbonding Period' in Cosmos Proof-of-Stake.", "expected_datasource": "rag"},
    {"question": "Describe the CurveCrypto Invariant and how it differs from the standard stableswap invariant.", "expected_datasource": "rag"},
    {"question": "How is the repegging condition (Loss < Profit / 2) calculated in CurveCrypto pools?", "expected_datasource": "rag"},
    {"question": "What is the LLAMMA mechanism in Liquity V2?", "expected_datasource": "rag"},
    {"question": "How does the PegKeeper contract help stabilize the BOLD stablecoin?", "expected_datasource": "rag"},
    {"question": "Compare Frontrunning and Backrunning in the context of MEV.", "expected_datasource": "rag"},
    {"question": "According to the Uniswap V2 docs, how does the TWAP oracle resist manipulation?", "expected_datasource": "rag"},
    {"question": "What are Flash Swaps in Uniswap V2 and how do they work?", "expected_datasource": "rag"},
    {"question": "Explain the concept of 'Concentrated Liquidity' in Uniswap V3.", "expected_datasource": "rag"},
    {"question": "How are 'Ticks' used to manage price ranges in Uniswap V3?", "expected_datasource": "rag"},
    {"question": "What is the role of Maker Vaults in generating Dai?", "expected_datasource": "rag"},
    {"question": "What distinguishes Multi-Collateral Dai (MCD) from Single-Collateral Dai (SCD)?", "expected_datasource": "rag"},
    {"question": "Describe the 'Group Law' for point addition on elliptic curves.", "expected_datasource": "rag"}, # Specific ECC concept
    {"question": "Explain the State Transition System model used in Ethereum.", "expected_datasource": "rag"}, # Foundational concept from Yellow Paper keywords

    # --- Web Search Examples (Real-time, News, Outside Corpus) ---
    {"question": "What is the current price of Chainlink (LINK)?", "expected_datasource": "web_search"},
    {"question": "Show me the latest news regarding SEC regulation of cryptocurrencies.", "expected_datasource": "web_search"},
    {"question": "What are the current Ethereum gas fees?", "expected_datasource": "web_search"},
    {"question": "Tell me about the Polkadot blockchain ecosystem.", "expected_datasource": "web_search"}, # Assumed outside the specific corpus keywords
    {"question": "Who won the most recent ETHGlobal hackathon?", "expected_datasource": "web_search"},
    {"question": "What is the market sentiment around Cardano today?", "expected_datasource": "web_search"},
    {"question": "Are there any major upcoming upgrades for the Avalanche network?", "expected_datasource": "web_search"}, # Assumed outside keywords
    {"question": "Find recent analyst predictions for Bitcoin's price next quarter.", "expected_datasource": "web_search"},
    {"question": "What is the total value locked (TVL) in DeFi right now?", "expected_datasource": "web_search"},
    {"question": "Compare the transaction speeds of Solana vs Aptos.", "expected_datasource": "web_search"}, # Aptos likely outside keywords

    # --- LLM Examples (General Knowledge, Conversational, Basic Concepts) ---
    {"question": "Hello, how are you today?", "expected_datasource": "llm"},
    {"question": "What is a blockchain?", "expected_datasource": "llm"}, # Generic explanation
    {"question": "Explain Proof-of-Stake in simple terms.", "expected_datasource": "llm"}, # Generic explanation, not tied to specific doc detail
    {"question": "Write a python function to calculate a moving average.", "expected_datasource": "llm"},
    {"question": "Tell me a short story about a time-traveling crypto trader.", "expected_datasource": "llm"},
    {"question": "What is the capital of France?", "expected_datasource": "llm"},
    {"question": "What does 'DeFi' stand for?", "expected_datasource": "llm"},
    {"question": "Can you help me debug this solidity code snippet? `contract Simple { ... }`", "expected_datasource": "llm"},
    {"question": "Summarize the concept of NFTs.", "expected_datasource": "llm"},
    {"question": "What is 5 factorial?", "expected_datasource": "llm"},
    {"question": "What are the main differences between Bitcoin and Ethereum?", "expected_datasource": "llm"}, # General comparison
    {"question": "Give me ideas for naming a new crypto project.", "expected_datasource": "llm"},
    {"question": "Who is Vitalik Buterin?", "expected_datasource": "llm"},
    {"question": "Translate 'decentralized finance' into Spanish.", "expected_datasource": "llm"},
    {"question": "Thanks for the help!", "expected_datasource": "llm"},
]


# --- Model and Component Setup ---

# Renamed variables for clarity/convention
llm_model = None
embedding_model = None
rag_classifier = None
is_classifier_available = False
vector_store = None
langgraph_app = None

# Check for API Keys - Initialize these before potentially using them
try:
    if not GEMINI_API_KEY:
        logging.error("GEMINI_API_KEY not set.")
        # Display error in UI if this were a Streamlit app, logging is appropriate here
    if not SERPER_API_KEY:
        logging.warning("SERPER_API_KEY not set. Web search will be disabled.")

    # LLM Initialization
    llm_model = ChatDeepSeek(model="deepseek-chat",api_key=DEEPSEEK_API)
    #llm_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=GEMINI_API_KEY)
    logging.info("Google Generative AI Chat Model initialized.")

    # Embedding Function Initialization
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY,
        task_type="RETRIEVAL_QUERY"
    )
    logging.info("Google Generative AI Embeddings initialized.")

    # Classifier Initialization
    if torch.cuda.is_available():
        classifier_device_index = 0
        logging.info("GPU detected. Using GPU device: 0 for classifier.")
    else:
        classifier_device_index = -1
        logging.info("No GPU detected. Using CPU for classifier.")

    rag_classifier = pipeline(
        task="text-classification",
        model="TBM99/Router-RAG-v2", # Using the specified model
        tokenizer="answerdotai/ModernBERT-base", # Using the specified tokenizer
        device=classifier_device_index,
    )
    _ = rag_classifier("warmup query") # Warm-up run
    logging.info("Text classification pipeline (router) loaded successfully.")
    is_classifier_available = True

except Exception as e:
    logging.error(f"Core component initialization Error: {e}", exc_info=True)
    # Components that failed will remain None


# RAG Setup - Attempt even if other components failed, log errors
try:
    if os.path.exists(CHROMA_DB_PATH) and embedding_model:
         chroma_persistent_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
         # Renamed variable
         available_collection_names = [col.name for col in chroma_persistent_client.list_collections()]

         if CHROMA_COLLECTION_NAME in available_collection_names:
            vector_store = Chroma(
                client=chroma_persistent_client,
                collection_name=CHROMA_COLLECTION_NAME,
                embedding_function=embedding_model, # Use the initialized embedding model
            )
            logging.info(f"ChromaDB vector store initialized from path '{CHROMA_DB_PATH}' with collection '{CHROMA_COLLECTION_NAME}'.")
         else:
             logging.warning(f"Chroma collection '{CHROMA_COLLECTION_NAME}' not found in path '{CHROMA_DB_PATH}'. RAG will be unavailable.")
             vector_store = None # Ensure it's None if collection not found
    elif not embedding_model:
        logging.warning(f"ChromaDB path '{CHROMA_DB_PATH}' exists, but embedding model failed to initialize. RAG disabled.")
        vector_store = None
    else:
        logging.warning(f"ChromaDB path '{CHROMA_DB_PATH}' not found. RAG disabled.")
        vector_store = None

except Exception as e:
    logging.error(f"ChromaDB Initialization failed: {e}", exc_info=True)
    vector_store = None # Ensure it's None on error

# --- LangGraph Workflow Definition ---
# Renamed workflow variable
graph_workflow = None

# Build graph only if essential components (LLM) are loaded
if llm_model:
    # Check if RAG and Classifier are available for full graph
    if vector_store and is_classifier_available:
        try:
            # Use functools.partial to pass dependencies to nodes
            # Assuming node functions are imported from funcs.py
            route_question_node_with_deps = functools.partial(route_question_node, classifier=rag_classifier)
            perform_rag_node_with_deps = functools.partial(perform_rag_node, vectorstore=vector_store, llm=llm_model)
            call_web_search_node_with_deps = functools.partial(call_web_search_node, serper_api_key=SERPER_API_KEY, llm=llm_model) # Pass key explicitly
            generate_llm_node_with_deps = functools.partial(generate_direct_llm_node, llm=llm_model)

            graph_workflow = StateGraph(GraphState) # Assuming GraphState is imported

            # Add nodes with clearer names matching function purpose
            graph_workflow.add_node("route_query", route_question_node_with_deps)
            graph_workflow.add_node("web_search", call_web_search_node_with_deps)
            graph_workflow.add_node("rag_retrieval_generation", perform_rag_node_with_deps)
            graph_workflow.add_node("direct_llm_generation", generate_llm_node_with_deps)

            graph_workflow.set_entry_point("route_query")
            graph_workflow.add_conditional_edges(
                "route_query",
                decide_next_node, # Assuming decide_next_node is imported
                {
                    # Map decision outcomes to node names
                    "call_web_search_node": "web_search", # Match the decision logic output
                    "perform_rag_node": "rag_retrieval_generation", # Match the decision logic output
                    "generate_direct_llm_node": "direct_llm_generation", # Match the decision logic output
                },
            )
            # Connect end points
            graph_workflow.add_edge("web_search", END)
            graph_workflow.add_edge("rag_retrieval_generation", END)
            graph_workflow.add_edge("direct_llm_generation", END)

            langgraph_app = graph_workflow.compile()
            logging.info("LangGraph workflow compiled successfully (Full RAG/Web/LLM).")

        except Exception as e:
            logging.error(f"LangGraph compilation failed (Full Graph): {e}", exc_info=True)
            langgraph_app = None # Ensure app is None on failure

    # Handle fallback if classifier or RAG failed, but LLM is available
    elif llm_model and (not is_classifier_available or not vector_store):
        try:
            logging.warning(f"Classifier available: {is_classifier_available}, Vector Store available: {bool(vector_store)}. Building graph with LLM-only or simplified routing.")

            # Define a fallback router if classifier is unavailable
            def llm_only_router_node(state: GraphState) -> dict:
                # Simple fallback: always route to LLM if classifier fails
                logging.warning("Classifier not available, defaulting route to direct LLM generation.")
                return {"datasource": "llm"} # Return value should match expected structure

            graph_workflow = StateGraph(GraphState) # Assuming GraphState is imported

            # Only define nodes that can run without the missing components
            generate_llm_node_with_deps = functools.partial(generate_direct_llm_node, llm=llm_model)
            graph_workflow.add_node("direct_llm_generation", generate_llm_node_with_deps)

            # Use fallback router if classifier is down, otherwise keep original router
            if not is_classifier_available:
                 graph_workflow.add_node("route_query", llm_only_router_node)
                 graph_workflow.set_entry_point("route_query")
                 # Simplified conditional edge directly to LLM node
                 graph_workflow.add_conditional_edges(
                     "route_query",
                     lambda state: "direct_llm_generation", # Always go to LLM
                     {"direct_llm_generation": "direct_llm_generation"}
                 )
            else:
                # If classifier is OK but RAG is not, the original router might still work but needs adjustment
                # For simplicity here, we'll route directly to LLM if RAG is unavailable but classifier worked.
                # A more robust solution would involve the router handling the 'rag' path failure.
                logging.warning("RAG vector store not available. Routing will default to LLM or Web Search if classifier chooses RAG.")
                # This part requires the original router logic (`route_question_node`) to gracefully handle
                # the RAG path being unavailable or modify the `decide_next_node` logic.
                # For this reproduction, we stick to the simpler LLM-only fallback if *either* fails.
                graph_workflow.add_node("route_query", llm_only_router_node) # Revert to simple LLM-only if RAG fails
                graph_workflow.set_entry_point("route_query")
                graph_workflow.add_conditional_edges(
                     "route_query",
                     lambda state: "direct_llm_generation",
                     {"direct_llm_generation": "direct_llm_generation"}
                 )


            graph_workflow.add_edge("direct_llm_generation", END)

            langgraph_app = graph_workflow.compile()
            logging.info("LangGraph workflow compiled (Fallback: LLM-only or simplified routing).")

        except Exception as e:
            logging.error(f"LangGraph compilation failed (Fallback Graph): {e}", exc_info=True)
            langgraph_app = None

else:
    logging.error("LLM model failed to initialize. Cannot build LangGraph workflow.")
    langgraph_app = None


# --- Backend Function ---
# Renamed function for clarity
def execute_graph_query(query: str) -> dict:
    """Invokes the compiled LangGraph app and returns the final state."""
    if not langgraph_app:
        logging.error("Attempted to run query, but LangGraph app is not compiled.")
        return {"error": "Graph not compiled due to initialization errors.", "question": query, "generation": "Error: Application backend not ready."}

    inputs = {"question": query}
    try:
        # Using invoke for synchronous execution
        final_state = langgraph_app.invoke(inputs, {"recursion_limit": 10})
        # Ensure final_state is a dict, handle potential None or other types
        if not isinstance(final_state, dict):
             logging.error(f"Graph invocation returned non-dict type: {type(final_state)}")
             return {"error": "Invalid response type from graph.", "question": query, "generation": "Error: Backend returned invalid data."}
        # Add question back into the final state for easier access if needed downstream
        final_state["question"] = query
        return final_state # Return the whole state dict

    except Exception as e:
        logging.error(f"Error during graph invocation for query '{query}': {e}", exc_info=True)
        return {"error": str(e), "question": query, "generation": f"An error occurred processing your request: {e}"}

# --- Define the Pipeline Runner Function ---
# This function will be called by LangSmith for each example in the dataset.
# Renamed function for clarity
def langsmith_pipeline_runner(inputs: dict) -> dict:
    """
    Runs the compiled LangGraph app for a given question input dictionary
    and returns outputs in the format expected by LangSmith evaluate.
    """
    question = inputs.get("question")
    if not question:
         logging.error("Pipeline runner received inputs without a 'question' key.")
         return {"generation": "Error: No question provided", "datasource": "error"}

    if not langgraph_app:
        logging.error("Pipeline runner called but LangGraph app is not compiled.")
        return {"generation": "Error: Backend not ready", "datasource": "error"}

    try:
        # Use invoke to get the final state easily
        final_state = langgraph_app.invoke({"question": question}, {"recursion_limit": 10})

        # Check if final_state is valid
        if not isinstance(final_state, dict):
            logging.error(f"Graph invocation in pipeline runner returned non-dict type: {type(final_state)}")
            return {"generation": "Error: Invalid backend response", "datasource": "error"}

        # Return the necessary outputs for evaluation, with robust .get() calls
        return {
            "generation": final_state.get("generation", "[No generation found]"),
            "datasource": final_state.get("datasource", "[No datasource found]"),
            # Include other state variables if needed for evaluation
        }
    except Exception as e:
        logging.error(f"Pipeline runner failed for question '{question}': {e}", exc_info=True)
        return {"generation": f"Error: {e}", "datasource": "error"}


# --- Define Custom Evaluators ---

# Evaluator 1: Check if the router chose the expected data source
# Renamed for clarity
@RunEvaluator
def evaluate_router_datasource_accuracy(run: Run, example: Example) -> EvaluationResult:
    """Checks if the final 'datasource' in the run outputs matches the expected one in the example."""
    if not example or not example.outputs or "expected_datasource" not in example.outputs:
        return EvaluationResult(key="router_datasource_accuracy", comment="Missing expected_datasource in example.")

    expected_datasource = example.outputs["expected_datasource"]

    # The pipeline_runner function returns the final state values as the run outputs
    if not run.outputs or "datasource" not in run.outputs:
        return EvaluationResult(key="router_datasource_accuracy", score=0, comment="Datasource not found in run outputs.")

    predicted_datasource = run.outputs["datasource"]

    # Handle potential errors or missing data in prediction
    if predicted_datasource == "[No datasource found]" or predicted_datasource == "error":
         score = 0
         comment = f"Predicted datasource was missing or error: {predicted_datasource}"
    else:
        score = 1 if predicted_datasource == expected_datasource else 0
        comment = f"Predicted: {predicted_datasource}, Expected: {expected_datasource}"

    return EvaluationResult(key="router_datasource_accuracy", score=score, comment=comment)


# Evaluator 2: Use LLM (Gemini) to judge relevance of the answer to the question
# Ensure GEMINI_API_KEY is available
# Renamed llm variable for clarity
llm_evaluator_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite", # Using flash for cost/speed, adjust if needed
    temperature=0,
    google_api_key=GEMINI_API_KEY,
    # Add client options if needed, e.g., for request retries
)
relevance_langchain_evaluator = load_evaluator("labeled_relevance", llm=llm_evaluator_model)

# Renamed for clarity
@RunEvaluator
async def evaluate_answer_relevance(run: Run, example: Example) -> EvaluationResult:
    """Checks if the generated answer is relevant to the input question using an LLM evaluator."""
    if not run.outputs or not run.outputs.get("generation") or run.outputs.get("generation") == "[No generation found]":
        return EvaluationResult(key="answer_relevance", score=0, comment="No valid generation found in run outputs.")

    # Ensure input question exists
    input_question = run.inputs.get("question")
    if not input_question:
        return EvaluationResult(key="answer_relevance", score=0, comment="Input question not found in run inputs.")

    prediction_text = run.outputs["generation"]
    # Handle potential errors passed as generation
    if prediction_text.startswith("Error:"):
         return EvaluationResult(key="answer_relevance", score=0, comment=f"Generation was an error message: {prediction_text}")

    try:
        eval_result_dict = await relevance_langchain_evaluator.aevaluate_strings(
            prediction=prediction_text,
            input=input_question,
            reference=example.outputs.get("reference_answer"), # Optional: include reference answer if available in dataset
        )
        # Ensure score is numeric, defaulting to 0 if not returned correctly
        raw_score = eval_result_dict.get("score", 0)
        if isinstance(raw_score, (int, float)):
            score = float(raw_score) # Ensure float for potential fractional scores
        else:
            score = 0.0 # Default to 0.0 if score is not a number (e.g., None or string)

        reasoning = eval_result_dict.get("reasoning", "No reasoning provided.")
        return EvaluationResult(key="answer_relevance", score=score, comment=reasoning)

    except Exception as e:
        logging.error(f"Answer Relevance evaluation failed for run {run.id}: {e}", exc_info=True)
        return EvaluationResult(key="answer_relevance", score=0, comment=f"Evaluation error: {e}")


# Evaluator 3: Check RAG Groundedness (Requires context extraction from trace)
# Renamed evaluator variable
groundedness_langchain_evaluator = load_evaluator("groundedness", llm=llm_evaluator_model)

# Renamed for clarity
@RunEvaluator
async def evaluate_rag_groundedness(run: Run, example: Example) -> EvaluationResult:
    """Checks if the RAG answer is grounded in the retrieved context (extracted from trace)."""
    # Only run this for runs where RAG was the chosen datasource
    run_datasource = run.outputs.get("datasource") if run.outputs else None
    if run_datasource != "rag": # Use the actual datasource value from the run
        return EvaluationResult(key="rag_groundedness", comment=f"Skipped: Run datasource was '{run_datasource}', not 'rag'.")

    generation_text = run.outputs.get("generation")
    if not generation_text or generation_text == "[No generation found]" or generation_text.startswith("Error:"):
        return EvaluationResult(key="rag_groundedness", score=0, comment="No valid generation found for groundedness check.")

    input_question = run.inputs.get("question")
    if not input_question:
        return EvaluationResult(key="rag_groundedness", score=0, comment="Input question not found in run inputs.")


    # --- Attempt to extract context from the trace ---
    # This depends heavily on the structure of your LangSmith trace.
    # Assuming the context is passed as input to the LLM call within the 'rag_retrieval_generation' node.
    retrieved_context_str = None
    try:
        # Look for the RAG node execution within the child runs of the main trace
        rag_node_run = None
        if run.child_runs:
            for child_run in run.child_runs:
                # Check the name of your RAG node as it appears in LangSmith traces (use the name defined in the graph)
                if child_run.name == "rag_retrieval_generation":
                    rag_node_run = child_run
                    break # Assume the first found is the relevant one

        if rag_node_run:
            # Check common places where context might be logged (adapt based on actual trace)
            # Option 1: In the direct inputs of the node itself (if logged)
            if rag_node_run.inputs and "context" in rag_node_run.inputs:
                retrieved_context_str = rag_node_run.inputs["context"]
            # Option 2: In the inputs to an LLM call *within* that node
            elif rag_node_run.child_llm_runs:
                 # Look at the inputs of the first LLM call within the RAG node
                 first_llm_run = rag_node_run.child_llm_runs[0]
                 if first_llm_run.inputs and first_llm_run.inputs.get("prompt"):
                      # This might require parsing the prompt, which is less reliable
                      # A better way is to ensure 'context' is explicitly logged somewhere
                      # Let's assume context is directly in inputs for now
                      logging.warning(f"Could not find 'context' directly in inputs of RAG node {rag_node_run.id}. Checking LLM prompt (less reliable).")
                      # Fallback: Check if context is logged elsewhere if the above fails
                      pass # Add more specific checks based on trace inspection

            # Ensure context is a string
            if retrieved_context_str and not isinstance(retrieved_context_str, str):
                retrieved_context_str = str(retrieved_context_str)


        if not retrieved_context_str:
            logging.warning(f"Could not find retrieved context in trace for RAG run {run.id}.")
            return EvaluationResult(key="rag_groundedness", score=0,
                                    comment="Could not find retrieved context in trace.")

        # Now run the groundedness check
        eval_result_dict = await groundedness_langchain_evaluator.aevaluate_strings(
            prediction=generation_text,
            context=retrieved_context_str,
            input=input_question # Providing question might help judge
        )
        # Groundedness evaluator often returns score=1 for grounded, 0 otherwise, or 'yes'/'no'
        raw_score = eval_result_dict.get("score")
        if isinstance(raw_score, (int, float)):
            score = float(raw_score)
        elif isinstance(raw_score, str): # Handle 'yes'/'no' string scores
             score = 1.0 if raw_score.strip().lower() == 'yes' else 0.0
        else:
             score = 0.0 # Default if score is missing or unexpected type

        reasoning = eval_result_dict.get("reasoning", "No reasoning provided.")
        return EvaluationResult(key="rag_groundedness", score=score, comment=reasoning)

    except Exception as e:
        logging.error(f"RAG Groundedness evaluation failed for run {run.id}: {e}", exc_info=True)
        return EvaluationResult(key="rag_groundedness", score=0, comment=f"Evaluation error: {e}")


# Evaluator 4: Check Web Search Groundedness (Similar logic to RAG groundedness)
# Renamed for clarity
@RunEvaluator
async def evaluate_web_search_groundedness(run: Run, example: Example) -> EvaluationResult:
    """Checks if the Web Search answer is grounded in the fetched web context (extracted from trace)."""
    run_datasource = run.outputs.get("datasource") if run.outputs else None
    # Use the actual datasource value from the run and the node name defined in the graph
    if run_datasource != "web_search":
        return EvaluationResult(key="web_search_groundedness", comment=f"Skipped: Run datasource was '{run_datasource}', not 'web_search'.")

    generation_text = run.outputs.get("generation")
    if not generation_text or generation_text == "[No generation found]" or generation_text.startswith("Error:"):
        return EvaluationResult(key="web_search_groundedness", score=0, comment="No valid generation found for groundedness check.")

    input_question = run.inputs.get("question")
    if not input_question:
        return EvaluationResult(key="web_search_groundedness", score=0, comment="Input question not found in run inputs.")

    # --- Attempt to extract context from the trace ---
    web_search_context_str = None
    try:
        web_search_node_run = None
        if run.child_runs:
            for child_run in run.child_runs:
                # Check the name of your Web Search node as defined in the graph
                if child_run.name == "web_search":
                    web_search_node_run = child_run
                    break

        if web_search_node_run:
            # The synthesis chain within the web search node likely receives context.
            # Check common places: node inputs or inputs to the internal LLM call.
            if web_search_node_run.inputs and "context" in web_search_node_run.inputs:
                web_search_context_str = web_search_node_run.inputs["context"]
            # Option 2: Check inputs of child LLM runs if context isn't in node inputs directly
            elif web_search_node_run.child_llm_runs:
                 first_llm_run = web_search_node_run.child_llm_runs[0]
                 if first_llm_run.inputs and first_llm_run.inputs.get("prompt"):
                      # Again, parsing prompt is less ideal. Explicit logging is better.
                      logging.warning(f"Could not find 'context' directly in inputs of Web Search node {web_search_node_run.id}. Checking LLM prompt (less reliable).")
                      pass # Add more specific checks based on trace inspection

            # Ensure context is a string
            if web_search_context_str and not isinstance(web_search_context_str, str):
                web_search_context_str = str(web_search_context_str)


        if not web_search_context_str:
            logging.warning(f"Could not find web context in trace for Web Search run {run.id}.")
            return EvaluationResult(key="web_search_groundedness", score=0,
                                    comment="Could not find web context in trace.")

        # Now run the groundedness check
        eval_result_dict = await groundedness_langchain_evaluator.aevaluate_strings(
            prediction=generation_text,
            context=web_search_context_str,
            input=input_question
        )
        # Handle scoring similar to RAG groundedness
        raw_score = eval_result_dict.get("score")
        if isinstance(raw_score, (int, float)):
            score = float(raw_score)
        elif isinstance(raw_score, str):
             score = 1.0 if raw_score.strip().lower() == 'yes' else 0.0
        else:
             score = 0.0

        reasoning = eval_result_dict.get("reasoning", "No reasoning provided.")
        return EvaluationResult(key="web_search_groundedness", score=score, comment=reasoning)

    except Exception as e:
        logging.error(f"Web Search Groundedness evaluation failed for run {run.id}: {e}", exc_info=True)
        return EvaluationResult(key="web_search_groundedness", score=0, comment=f"Evaluation error: {e}")


# --- Run the Evaluation ---
async def run_langsmith_evaluation():
    """Sets up and runs the LangSmith evaluation."""
    # Check if essential components are ready
    if not langgraph_app:
        logging.error("LangGraph app is not initialized. Cannot run evaluation.")
        print("ERROR: LangGraph app failed to initialize. Evaluation cancelled.")
        return
    if not llm_evaluator_model:
         logging.error("LLM Evaluator model is not initialized. Cannot run evaluations requiring it.")
         print("ERROR: LLM Evaluator model failed to initialize. Evaluation cancelled.")
         return

    langsmith_client = Client() # Assumes LANGCHAIN_API_KEY is set in environment or via os.environ

    # Optional: Create the dataset in LangSmith if it doesn't exist
    try:
        langsmith_client.read_dataset(dataset_name=DATASET_NAME)
        logging.info(f"Using existing LangSmith dataset: {DATASET_NAME}")
    except Exception: # Catch LangSmith specific exception if possible, otherwise general Exception
        logging.info(f"Creating LangSmith dataset: {DATASET_NAME}")
        try:
            langsmith_client.create_dataset(
                dataset_name=DATASET_NAME,
                description="Evaluation dataset for Crypto RAG/Web/LLM Router Bot V1 (Gemini)",
            )
            # Add examples to the dataset
            logging.info(f"Adding {len(EVALUATION_DATASET)} examples to dataset '{DATASET_NAME}'...")
            for example_data in EVALUATION_DATASET:
                langsmith_client.create_example(
                    inputs={"question": example_data["question"]},
                    outputs={"expected_datasource": example_data["expected_datasource"]}, # Add other outputs if needed later (e.g., reference answer)
                    dataset_name=DATASET_NAME,
                )
            logging.info(f"Finished adding examples to dataset '{DATASET_NAME}'.")
        except Exception as ds_creation_error:
            logging.error(f"Failed to create or populate LangSmith dataset '{DATASET_NAME}': {ds_creation_error}", exc_info=True)
            print(f"ERROR: Failed to create or populate LangSmith dataset '{DATASET_NAME}'. Evaluation cancelled.")
            return


    # Define the evaluators to run
    # Renamed list variable
    active_evaluators = [
        evaluate_router_datasource_accuracy,
        evaluate_answer_relevance
    ]

    print(f"\n--- Starting LangSmith Evaluation ---")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Project: {LANGSMITH_PROJECT_NAME}")
    print(f"Evaluators: {[e.__name__ for e in active_evaluators]}")

    # Run the evaluation using asyncio.run() for async evaluators
    # Renamed results variable
    evaluation_results = await evaluate(
        langsmith_pipeline_runner, # The function that runs the pipeline for one input
        data=DATASET_NAME,         # Name of the dataset in LangSmith or list of examples
        evaluators=active_evaluators,
        experiment_prefix="Crypto Bot Eval - Gemini Flash", # A name for this specific test run suite
        metadata={                 # Optional: Add metadata about the run configuration
            "pipeline_version": "1.1-corrected", # Example version
            "llm_model": llm_model.model_name if llm_model else "N/A",
            "embedding_model": embedding_model.model_name if embedding_model else "N/A",
            "router_model": rag_classifier.model.name_or_path if rag_classifier else "N/A",
            "chroma_collection": CHROMA_COLLECTION_NAME if vector_store else "N/A",
            "routing_mode": "Full" if is_classifier_available and vector_store else "Fallback",
        },
        max_concurrency=5, # Adjust based on API rate limits and desired speed
        # verbose=True # Uncomment for more detailed output during evaluation
    ) # evaluation_results will contain summary statistics

    print("\n--- Evaluation Complete ---")
    # The results object contains aggregated metrics. Print or process as needed.
    # print(evaluation_results) # You can print the summary results dict

    # Direct the user to the LangSmith project UI
    print(f"View detailed results in LangSmith project: {LANGSMITH_PROJECT_NAME}")
    # Constructing a potential URL (user needs to replace workspace/project if needed)
    ls_base_url = os.getenv("LANGCHAIN_ENDPOINT", "https://smith.langchain.com").replace("api.", "")
    print(f"Potential URL (check project name): {ls_base_url}/o/{os.getenv('LANGCHAIN_TENANT_ID','default')}/projects/p/{evaluation_results.project_name}")


# --- Main Execution Block ---
if __name__ == "__main__":
    # Logging is configured at the top now.

    # Check for necessary API keys (using the hardcoded ones for this script's logic)
    # We already logged warnings/errors during initialization if they were missing/problematic.
    # This check is slightly redundant given the hardcoding, but good practice if keys were from env.
    essential_components_ready = bool(llm_model and langgraph_app)

    if essential_components_ready:
        print("Initialization complete. Starting evaluation...")
        # Run the async evaluation function
        asyncio.run(run_langsmith_evaluation())
    else:
        print("ERROR: Essential components (LLM, LangGraph App) failed to initialize. Evaluation cannot proceed.")
        logging.critical("Evaluation cancelled due to initialization failures.")
