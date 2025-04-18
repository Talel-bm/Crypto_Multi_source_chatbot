# DeFi & Crypto Chatbot Project

This is a chatbot specialized in decentralized Finance (DeFi) and cryptocurrency. When given a query, it uses one of three methods:

1.  Searches the web for an answer.
2.  Performs Retrieval-Augmented Generation (RAG) using a knowledge base built from 20 selected documents (original PDFs are in the `"CRYPTO PAPERS"` folder).
3.  Lets the base Language Model (LLM) answer natively.

The techniques used in this project focus on scalability, meaning the system can be adapted to handle a larger number of documents and potentially more response options.

## Key Points:

*   **PDF Processing:** We used a high-performing and scalable method for converting PDF files to Markdown, combining **Marker** and **Gemini** (inspired by [community benchmarks](https://www.reddit.com/r/LocalLLaMA/comments/1jz80f1/i_benchmarked_7_ocr_solutions_on_a_complex/)).
*   **Text Chunking:** We employed the recursive chunker from the **Chonkie library** to segment the text effectively before indexing it into the vector store.
*   **Intelligent Routing:** We [fine-tuned a **ModernBERT**](https://huggingface.co/TBM99/Router-RAG-v2) model to act as a router between the `"llm"`, `"rag"`, and `"web_search"` options.
    *   This router was trained on a synthetic dataset (generated with Gemini) focused on specific **keywords** found within the source documents.
    *   Its purpose is to direct queries to the RAG system *only* when the answer likely requires information from those specific documents.
    *   Any unrelated question, even if related to DeFi/crypto in general, is directed to the base LLM.
    *   The fine-tuned router achieves a performant F1 score of **0.85**.
The implementations for these key components reside in separate folders within the project.
* **Evaluation with LangSmith** : we evaluated the pipeline with LangSmith and provided screenshots of the results The results were great for routing correctness and answer relevance, the latency however highly depended on the LLM, with good latency with Google flash models and slower response with Deepseek-chat.

The main `app.py` file serves as both the backend logic orchestrator and the frontend for the Streamlit chatbot interface.

The main AI tool that was used for the development of this project was GOOGLE's AI-studio interface.
