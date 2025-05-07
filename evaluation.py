# --- Filename: langsmith_evaluation_refactored.py ---
import os
import asyncio # Needed for async evaluators
import logging
from langsmith import Client
from langsmith.schemas import Example, Run  # Base schemas
from langsmith.evaluation import EvaluationResult # Evaluation-specific schema
from langchain.evaluation import load_evaluator, EvaluatorType
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_deepseek import ChatDeepSeek
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
import chromadb
from transformers import pipeline
import torch
import functools
import time
from dotenv import load_dotenv

# --- IMPORTANT: Assumes funcs.py exists in the same directory ---
# It must define: GraphState, route_question_node, perform_rag_node,
# call_web_search_node, generate_direct_llm_node, decide_next_node
try:
    from funcs import *
except ImportError:
    print("ERROR: Could not import from funcs.py.")
    print("Please ensure funcs.py exists in the same directory and defines:")
    print("- GraphState class/TypedDict")
    print("- route_question_node, perform_rag_node, call_web_search_node, generate_direct_llm_node")
    print("- decide_next_node")
    exit()

# --- Configuration & Initialization ---
# WARNING: Hardcoding keys is insecure. Use environment variables.
LANGSMITH_TRACING = True
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
DEEPSEEK_API = os.getenv("DEEPSEEK_API", "")

CHROMA_DB_PATH = "./DOC DB" # Assumes this directory exists and has the DB
CHROMA_COLLECTION_NAME = "crypto_db"
LANGSMITH_PROJECT_NAME = os.getenv("LANGCHAIN_PROJECT", "Crypto Bot Evaluation")
DATASET_NAME = "Crypto Bot Eval Dataset V1" # Or choose a new name

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set LangChain environment variables if not already set
if "LANGSMITH_API_KEY" not in os.environ and LANGSMITH_API_KEY and LANGSMITH_API_KEY != "YOUR_LANGSMITH_API_KEY":
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    logging.info("Set LANGSMITH_API_KEY from script variable.")
elif "LANGSMITH_API_KEY" not in os.environ:
    logging.warning("LANGSMITH_API_KEY environment variable not set.")

if "LANGCHAIN_TRACING_V2" not in os.environ:
     os.environ["LANGCHAIN_TRACING_V2"] = "true" # Enable tracing

if "LANGCHAIN_PROJECT" not in os.environ and LANGSMITH_PROJECT_NAME:
     os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT_NAME
     logging.info(f"Set LANGCHAIN_PROJECT to '{LANGSMITH_PROJECT_NAME}' from script variable.")
elif "LANGCHAIN_PROJECT" not in os.environ:
    logging.warning("LANGCHAIN_PROJECT environment variable not set.")


# --- Evaluation Dataset ---
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
llm_model = None
embedding_model = None
rag_classifier = None
is_classifier_available = False
vector_store = None
langgraph_app = None

try:
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        logging.error("GEMINI_API_KEY not set or is placeholder.")
    else:
        llm_model = ChatDeepSeek(model="deepseek-chat", api_key=DEEPSEEK_API)
        #llm_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=GEMINI_API_KEY)
        logging.info("Google Generative AI Chat Model initialized.")
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=GEMINI_API_KEY, task_type="RETRIEVAL_QUERY"
        )
        logging.info("Google Generative AI Embeddings initialized.")

    if not SERPER_API_KEY or SERPER_API_KEY == "YOUR_SERPER_API_KEY":
        logging.warning("SERPER_API_KEY not set or is placeholder. Web search may be disabled in graph nodes.")

    # Classifier Initialization
    try:
        if torch.cuda.is_available():
            classifier_device_index = 0
            logging.info("GPU detected. Using GPU device: 0 for classifier.")
        else:
            classifier_device_index = -1
            logging.info("No GPU detected. Using CPU for classifier.")

        rag_classifier = pipeline(
            task="text-classification", model="TBM99/Router-RAG-v2",
            tokenizer="answerdotai/ModernBERT-base", device=classifier_device_index,
        )
        _ = rag_classifier("warmup query")
        logging.info("Text classification pipeline (router) loaded successfully.")
        is_classifier_available = True
    except Exception as classifier_err:
         logging.error(f"Failed to initialize classifier: {classifier_err}", exc_info=True)
         is_classifier_available = False

except Exception as core_init_err:
    logging.error(f"Core component initialization Error: {core_init_err}", exc_info=True)

# RAG Setup
try:
    if os.path.exists(CHROMA_DB_PATH) and embedding_model:
         chroma_persistent_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
         available_collection_names = [col.name for col in chroma_persistent_client.list_collections()]
         if CHROMA_COLLECTION_NAME in available_collection_names:
            vector_store = Chroma(
                client=chroma_persistent_client,
                collection_name=CHROMA_COLLECTION_NAME,
                embedding_function=embedding_model,
            )
            logging.info(f"ChromaDB vector store initialized from '{CHROMA_DB_PATH}' with collection '{CHROMA_COLLECTION_NAME}'.")
         else:
             logging.warning(f"Chroma collection '{CHROMA_COLLECTION_NAME}' not found in '{CHROMA_DB_PATH}'. RAG unavailable.")
             vector_store = None
    elif not embedding_model:
        logging.warning(f"ChromaDB path '{CHROMA_DB_PATH}' may exist, but embedding model failed. RAG disabled.")
        vector_store = None
    else:
        logging.warning(f"ChromaDB path '{CHROMA_DB_PATH}' not found. RAG disabled.")
        vector_store = None
except Exception as db_err:
    logging.error(f"ChromaDB Initialization failed: {db_err}", exc_info=True)
    vector_store = None

# --- LangGraph Workflow Definition ---
graph_workflow = None
if llm_model:
    if vector_store and is_classifier_available:
        try:
            # Use functools.partial or ensure nodes access components appropriately
            route_question_node_with_deps = functools.partial(route_question_node, classifier=rag_classifier)
            perform_rag_node_with_deps = functools.partial(perform_rag_node, vectorstore=vector_store, llm=llm_model)
            # Ensure call_web_search_node gets API key if needed
            call_web_search_node_with_deps = functools.partial(call_web_search_node, SERPER_API_KEY=SERPER_API_KEY, llm=llm_model)
            generate_llm_node_with_deps = functools.partial(generate_direct_llm_node, llm=llm_model)

            graph_workflow = StateGraph(GraphState)
            graph_workflow.add_node("route_query", route_question_node_with_deps)
            graph_workflow.add_node("web_search", call_web_search_node_with_deps)
            graph_workflow.add_node("rag_retrieval_generation", perform_rag_node_with_deps)
            graph_workflow.add_node("direct_llm_generation", generate_llm_node_with_deps)
            graph_workflow.set_entry_point("route_query")
            graph_workflow.add_conditional_edges(
                "route_query", decide_next_node, {
                    "call_web_search_node": "web_search",
                    "perform_rag_node": "rag_retrieval_generation",
                    "generate_direct_llm_node": "direct_llm_generation",
                }
            )
            graph_workflow.add_edge("web_search", END)
            graph_workflow.add_edge("rag_retrieval_generation", END)
            graph_workflow.add_edge("direct_llm_generation", END)
            langgraph_app = graph_workflow.compile()
            logging.info("LangGraph workflow compiled successfully (Full RAG/Web/LLM).")
        except Exception as graph_err:
            logging.error(f"LangGraph compilation failed (Full Graph): {graph_err}", exc_info=True)
            langgraph_app = None
    else: # Fallback logic
        try:
            logging.warning(f"Classifier available: {is_classifier_available}, Vector Store available: {bool(vector_store)}. Building graph with fallback routing.")
            def llm_only_router_node(state: GraphState) -> dict:
                logging.warning("Fallback router: defaulting route to direct LLM generation.")
                # Your router might have more sophisticated fallback logic
                return {"datasource": "llm"}

            graph_workflow = StateGraph(GraphState)
            generate_llm_node_with_deps = functools.partial(generate_direct_llm_node, llm=llm_model)
            graph_workflow.add_node("direct_llm_generation", generate_llm_node_with_deps)

            # Always use fallback if critical components missing for full routing
            graph_workflow.add_node("route_query", llm_only_router_node)
            graph_workflow.set_entry_point("route_query")
            graph_workflow.add_conditional_edges(
                 "route_query", lambda state: "direct_llm_generation",
                 {"direct_llm_generation": "direct_llm_generation"}
             )
            graph_workflow.add_edge("direct_llm_generation", END)
            langgraph_app = graph_workflow.compile()
            logging.info("LangGraph workflow compiled (Fallback routing).")
        except Exception as fallback_graph_err:
            logging.error(f"LangGraph compilation failed (Fallback Graph): {fallback_graph_err}", exc_info=True)
            langgraph_app = None
else:
    logging.error("LLM model failed to initialize. Cannot build LangGraph workflow.")
    langgraph_app = None

# --- Define the Target Function for LangSmith Evaluation ---
def pipeline_target_for_evaluation(inputs: dict) -> dict:
    """
    Runs the compiled LangGraph app for a given input question and returns
    a dictionary containing generation, datasource, and retrieved context.
    """
    question = inputs.get("question")
    output = {
        "generation": "[Error: Processing failed]",
        "datasource": "error",
        "retrieved_context": None,
    }

    if not question:
        logging.error("Target function received inputs without a 'question' key.")
        output["generation"] = "[Error: No question provided]"
        return output

    if not langgraph_app:
        logging.error("Target function called but LangGraph app is not compiled.")
        output["generation"] = "[Error: Backend not ready]"
        return output

    try:
        final_state = langgraph_app.invoke({"question": question}, {"recursion_limit": 10})

        if not isinstance(final_state, dict):
            logging.error(f"Graph invocation returned non-dict type: {type(final_state)}")
            output["generation"] = "[Error: Invalid backend response]"
            return output

        generation = final_state.get("generation", "[No generation found]")
        datasource = final_state.get("datasource", "[No datasource found]")
        retrieved_context_str = None

        # --- CRITICAL: Adapt context extraction based on your ACTUAL graph state keys ---
        # Check the keys used in GraphState by perform_rag_node and call_web_search_node
        # Example keys (replace with your actual keys):
        rag_docs_key = "documents"          # Key holding list of LangChain docs after RAG
        web_results_key = "web_search_results" # Key holding web search result strings/dicts

        if datasource == "rag":
            retrieved_context_str = output.get('retrieved_context')


        output["generation"] = generation
        output["datasource"] = datasource
        output["retrieved_context"] = retrieved_context_str

    except Exception as e:
        logging.error(f"Target function failed for question '{question}': {e}", exc_info=True)
        output["generation"] = f"[Error: {e}]"
        output["datasource"] = "error"

    return output


# --- Define Custom Evaluator Functions ---

# Evaluator 1: Router Datasource Accuracy
def evaluate_router_datasource_accuracy(inputs: dict, outputs: dict, reference_outputs: dict) -> EvaluationResult:
    """Checks if the 'datasource' in outputs matches 'expected_datasource' in reference_outputs."""
    key = "router_datasource_accuracy"
    expected_datasource = reference_outputs.get("expected_datasource")
    if not expected_datasource:
        return EvaluationResult(key=key, comment="Missing 'expected_datasource' in reference outputs.")

    predicted_datasource = outputs.get("datasource")
    score = 0.0
    if not predicted_datasource or predicted_datasource == "error":
        comment = f"Predicted datasource was missing or error: {predicted_datasource}. Expected: {expected_datasource}"
    elif predicted_datasource == "[No datasource found]":
        comment = f"Predicted datasource was not found. Expected: {expected_datasource}"
    else:
        score = 1.0 if predicted_datasource == expected_datasource else 0.0
        comment = f"Predicted: {predicted_datasource}, Expected: {expected_datasource}"

    return EvaluationResult(key=key, score=score, comment=comment)

# Evaluator 2: Answer Relevance (Async)
try:
    llm_evaluator_model_relevance = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite", temperature=0, google_api_key=GEMINI_API_KEY
    ) if llm_model else None
    # --- CHANGE HERE ---
    relevance_langchain_evaluator = (
        load_evaluator(
            EvaluatorType.CRITERIA,  # the “criteria” evaluator
            llm=llm_evaluator_model_relevance,
            criteria="relevance",  # judge by the “relevance” criterion
        ) if llm_evaluator_model_relevance else None)
    # --- END CHANGE ---
except Exception as relevance_init_err:
    logging.error(f"Failed to initialize relevance evaluator: {relevance_init_err}", exc_info=True)
    relevance_langchain_evaluator = None

def evaluate_answer_relevance(inputs: dict, outputs: dict, reference_outputs: dict) -> EvaluationResult:
    """Checks if the generated answer is relevant to the input question using the 'criteria' evaluator."""
    key = "answer_relevance"
    if not relevance_langchain_evaluator:
        return EvaluationResult(key=key, score=0, comment="Relevance evaluator not initialized.")

    prediction_text = outputs.get("generation")
    input_question = inputs.get("question")

    if not prediction_text or prediction_text.startswith("[Error:") or prediction_text == "[No generation found]":
        return EvaluationResult(key=key, score=0, comment="Invalid or missing generation.")
    if not input_question:
        return EvaluationResult(key=key, score=0, comment="Missing 'question' in inputs.")

    try:
        # The 'criteria' evaluator primarily uses input and prediction.
        # Reference answer isn't directly used by the default 'relevance' criterion prompt,
        # but it's good practice to keep it if you customize prompts later.
        reference_answer = reference_outputs.get("answer") # Keep for potential future use

        # --- MINOR CHANGE MIGHT BE NEEDED HERE depending on exact langchain version ---
        # Pass input and prediction. The 'criteria' evaluator knows what to do based on initialization.
        eval_result_dict = relevance_langchain_evaluator.evaluate_strings(
            prediction=prediction_text,
            input=input_question,
            # reference=reference_answer, # Reference might not be directly used by default relevance criteria
        )
        # --- END MINOR CHANGE ---

        # Score interpretation might change slightly based on criteria prompts.
        # Often returns 'Y'/'N' or 1/0. Assume 1 for 'Y' / pass, 0 otherwise.
        raw_score = eval_result_dict.get("score")
        score = 0.0
        if isinstance(raw_score, (int, float)):
            score = float(raw_score)
        elif isinstance(raw_score, str):
            # Handle 'Y'/'N' or other string outputs if the LLM returns them
            score = 1.0 if raw_score.strip().upper() == 'Y' else 0.0
        else: # Default to 0 if score is missing or unexpected type
             score = 0.0

        reasoning = eval_result_dict.get("reasoning", "No reasoning provided.")
        return EvaluationResult(key=key, score=score, comment=reasoning)

    except Exception as e:
        logging.error(f"Answer Relevance evaluation failed: {e}", exc_info=True)
        return EvaluationResult(key=key, score=0, comment=f"Evaluation error: {e}")

try:
    llm_evaluator_model_groundedness = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite", temperature=0, google_api_key=GEMINI_API_KEY
    ) if llm_model else None
    # --- CHANGE HERE ---
    groundedness_langchain_evaluator = (
        load_evaluator(
            EvaluatorType.LABELED_CRITERIA,
            llm=llm_evaluator_model_groundedness,
            criteria="correctness",  # one of the built‑ins
        )
        if llm_evaluator_model_groundedness
        else None)
    # --- END CHANGE ---
except Exception as groundedness_init_err:
    logging.error(f"Failed to initialize groundedness evaluator: {groundedness_init_err}", exc_info=True)
    groundedness_langchain_evaluator = None

def evaluate_groundedness(inputs: dict, outputs: dict, reference_outputs: dict) -> EvaluationResult:
    """Checks if the answer is grounded in the retrieved context using the 'criteria' evaluator."""
    datasource = outputs.get("datasource")
    key = "groundedness_skipped"
    if datasource == "rag": key = "rag_groundedness"
    elif datasource == "web_search": key = "web_search_groundedness"
    else: return EvaluationResult(key=key, comment=f"Skipped: Datasource is '{datasource}'.")

    if not groundedness_langchain_evaluator:
         return EvaluationResult(key=key, score=0, comment="Groundedness evaluator not initialized.")

    generation_text = outputs.get("generation")
    input_question = inputs.get("question")
    retrieved_context_str = outputs.get("context")

    if not generation_text or generation_text.startswith("[Error:") or generation_text == "[No generation found]":
        return EvaluationResult(key=key, score=0, comment="Invalid or missing generation.")
    if not input_question:
        return EvaluationResult(key=key, score=0, comment="Missing 'question' in inputs.")
    if not retrieved_context_str:
        logging.warning(f"Groundedness check for {datasource} run received no context.")
        return EvaluationResult(key=key, score=0, comment="No retrieved context provided by target function.")

    try:
        # --- CHANGE HERE ---
        # The 'groundedness' criteria evaluator needs prediction and context. Input is often helpful too.
        eval_result_dict = groundedness_langchain_evaluator.aevaluate_strings(
            prediction=generation_text,
            context=retrieved_context_str,
            input=input_question # Provide input question for better context judgment
        )
        # --- END CHANGE ---

        # Handle scoring similar to relevance (often Y/N or 1/0)
        raw_score = eval_result_dict.get("score")
        score = 0.0
        if isinstance(raw_score, (int, float)): score = float(raw_score)
        elif isinstance(raw_score, str): score = 1.0 if raw_score.strip().upper() == 'Y' else 0.0
        reasoning = eval_result_dict.get("reasoning", "No reasoning provided.")
        return EvaluationResult(key=key, score=score, comment=reasoning)

    except Exception as e:
        logging.error(f"{key} evaluation failed: {e}", exc_info=True)
        return EvaluationResult(key=key, score=0, comment=f"Evaluation error: {e}")

# --- Run the Evaluation ---
def run_langsmith_evaluation():
    """Sets up and runs the LangSmith evaluation using client.evaluate."""

    if not langgraph_app:
        logging.error("LangGraph app is not initialized. Cannot run evaluation.")
        print("ERROR: LangGraph app failed to initialize. Evaluation cancelled.")
        return
    if not (llm_evaluator_model_relevance and groundedness_langchain_evaluator):
         logging.error("One or more LLM-based evaluators failed to initialize.")
         # Decide if you want to proceed with only the accuracy evaluator or stop
         # For now, we'll stop if critical evaluators are missing.
         print("ERROR: LLM Evaluators failed to initialize. Evaluation cancelled.")
         return

    try:
        langsmith_client = Client() # Assumes LANGSMITH_API_KEY is set
    except Exception as client_err:
        logging.error(f"Failed to initialize LangSmith client: {client_err}", exc_info=True)
        print(f"ERROR: Failed to initialize LangSmith client. Check API key and connectivity.")
        return

    # Dataset Check/Creation
    try:
        langsmith_client.read_dataset(dataset_name=DATASET_NAME)
        logging.info(f"Using existing LangSmith dataset: {DATASET_NAME}")
    except Exception:
        logging.info(f"Creating LangSmith dataset: {DATASET_NAME}")
        try:
            dataset = langsmith_client.create_dataset(
                dataset_name=DATASET_NAME,
                description="Evaluation dataset for Crypto RAG/Web/LLM Router Bot (Refactored Eval)",
            )
            logging.info(f"Adding {len(EVALUATION_DATASET)} examples to dataset '{DATASET_NAME}'...")
            examples_to_create = [
                {
                    "inputs": {"question": ex_data["question"]},
                    "outputs": {"expected_datasource": ex_data["expected_datasource"]},
                    # Add "answer": "..." to outputs if you have reference answers
                 } for ex_data in EVALUATION_DATASET
            ]
            langsmith_client.create_examples(dataset_id=dataset.id, examples=examples_to_create)
            logging.info(f"Finished adding examples to dataset '{DATASET_NAME}'.")
        except Exception as ds_creation_error:
            logging.error(f"Failed to create or populate LangSmith dataset '{DATASET_NAME}': {ds_creation_error}", exc_info=True)
            print(f"ERROR: Failed to create or populate LangSmith dataset '{DATASET_NAME}'. Evaluation cancelled.")
            return

    # Define Active Evaluators
    active_evaluators = [
        evaluate_router_datasource_accuracy,
        evaluate_answer_relevance,
        evaluate_groundedness,
    ]
    # Filter out evaluators that failed to initialize
    active_evaluators = [
        ev for ev in active_evaluators
        if not (ev.__name__ == 'evaluate_answer_relevance' and not relevance_langchain_evaluator) and \
           not (ev.__name__ == 'evaluate_groundedness' and not groundedness_langchain_evaluator)
    ]
    if not active_evaluators:
        print("ERROR: No active evaluators available. Evaluation cancelled.")
        return

    print(f"\n--- Starting LangSmith Evaluation ---")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Project: {os.environ.get('LANGCHAIN_PROJECT', 'Not Set')}")
    print(f"Evaluators: {[e.__name__ for e in active_evaluators]}")

    # Run Evaluation using client.evaluate
    try:
        experiment_results = langsmith_client.evaluate(
            pipeline_target_for_evaluation,
            data=DATASET_NAME,
            evaluators=active_evaluators,
            experiment_prefix="Crypto Bot Eval - Refactored",
            metadata={
                "pipeline_version": "1.2-refactored",
                "llm_model": llm_model.model_name if llm_model and hasattr(llm_model, 'model_name') else "N/A",
                "embedding_model": embedding_model.model_name if embedding_model and hasattr(embedding_model, 'model_name') else "N/A",
                "router_model": rag_classifier.model.name_or_path if rag_classifier and hasattr(rag_classifier, 'model') and hasattr(rag_classifier.model, 'name_or_path') else "N/A",
                "chroma_collection": CHROMA_COLLECTION_NAME if vector_store else "N/A",
                "routing_mode": "Full" if is_classifier_available and vector_store else "Fallback",
            },
            max_concurrency=5,
        )

        print("\n--- Evaluation Complete ---")
        project_name = experiment_results.experiment_prefix
        print(f"View detailed results in LangSmith project: {project_name}")
        ls_base_url = os.getenv("LANGCHAIN_ENDPOINT", "https://smith.langchain.com").replace("api.", "")
        tenant_handle = os.getenv('LANGCHAIN_TENANT_ID', 'default')
        # Constructing the URL more reliably
        project_link = f"{ls_base_url}/o/{tenant_handle}/projects/p/{project_name}"
        print(f"View results at: {project_link}")

    except Exception as eval_run_err:
         logging.error(f"LangSmith evaluation run failed: {eval_run_err}", exc_info=True)
         print(f"ERROR: LangSmith evaluation run failed. Check logs and LangSmith UI for details.")

# --- Main Execution Block ---
if __name__ == "__main__":
    # Check essential components initialized successfully
    essential_components_ready = bool(llm_model and langgraph_app)

    if essential_components_ready:
        print("Initialization complete. Starting evaluation...")
        run_langsmith_evaluation() # Now synchronous
    else:
        print("ERROR: Essential components (LLM, LangGraph App) failed to initialize. Evaluation cannot proceed.")
        logging.critical("Evaluation cancelled due to initialization failures.")
