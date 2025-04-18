import http.client
import json
import os
import operator
from typing import TypedDict, Literal, List
import logging
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import chromadb.utils.embedding_functions as embedding_functions
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb
from langgraph.graph import StateGraph, END

def router(query,classifier):
    decision = {0:"web_search",1:"llm",2:"rag"}
    return decision[classifier(query)[0]['label']]
def format_search_results(search_response):
    try:
        # Extract search query
        search_query = search_response.get('searchParameters', {}).get('q', 'Unknown query')

        # Start building the formatted output
        formatted_output = f"# Search Results for: {search_query}\n\n"

        # Add answer box if present
        answer_box = search_response.get('answerBox', {})
        if answer_box:
            formatted_output += "## Featured Answer\n"
            title = answer_box.get('title', '')
            answer = answer_box.get('answer', '')
            source = answer_box.get('source', '')

            if title:
                formatted_output += f"**{title}** "

            if answer:
                formatted_output += f"{answer}\n"

            if source:
                formatted_output += f"Source: {source}\n"

            formatted_output += "\n"

        # Add organic search results
        organic_results = search_response.get('organic', [])
        if organic_results:
            formatted_output += "## Top Search Results\n\n"

            for i, result in enumerate(organic_results, 1):
                title = result.get('title', 'No title')
                link = result.get('link', 'No link')
                snippet = result.get('snippet', 'No description')
                rating = result.get('rating', None)
                rating_count = result.get('ratingCount', None)

                formatted_output += f"### {i}. {title}\n"
                formatted_output += f"**Link:** {link}\n"
                formatted_output += f"**Description:** {snippet}\n"

                if rating is not None and rating_count is not None:
                    formatted_output += f"**Rating:** {rating}/5 ({rating_count} reviews)\n"
                elif rating is not None:
                    formatted_output += f"**Rating:** {rating}/5\n"

                # Add sitelinks if present
                sitelinks = result.get('sitelinks', [])
                if sitelinks:
                    formatted_output += "\n**Related Links:**\n"
                    for sitelink in sitelinks:
                        sitelink_title = sitelink.get('title', 'No title')
                        sitelink_link = sitelink.get('link', 'No link')
                        formatted_output += f"- [{sitelink_title}]({sitelink_link})\n"

                formatted_output += "\n"

        # Add "People Also Ask" section
        people_also_ask = search_response.get('peopleAlsoAsk', [])
        if people_also_ask:
            formatted_output += "## People Also Ask\n\n"

            for i, question_data in enumerate(people_also_ask, 1):
                question = question_data.get('question', 'Unknown question')
                snippet = question_data.get('snippet', 'No answer available')
                title = question_data.get('title', '')
                link = question_data.get('link', '')

                formatted_output += f"### Q{i}: {question}\n"
                formatted_output += f"{snippet}\n"

                if title and link:
                    formatted_output += f"**Source:** [{title}]({link})\n"

                formatted_output += "\n"

        # Add related searches
        related_searches = search_response.get('relatedSearches', [])
        if related_searches:
            formatted_output += "## Related Searches\n\n"

            for i, related in enumerate(related_searches, 1):
                query = related.get('query', 'Unknown query')
                formatted_output += f"- {query}\n"

        # Footer with credits info
        credits = search_response.get('credits', 'Unknown')
        formatted_output += f"\n*Search powered by Google via Serper API. Credits used: {credits}*"

        return formatted_output

    except Exception as e:
        return f"Error formatting search results: {str(e)}"
def web_search(query, SERPER_API_KEY):
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({"q": query})
    headers = {
      'X-API-KEY': SERPER_API_KEY,
      'Content-Type': 'application/json'
    }
    conn.request("POST", "/search", payload, headers)
    res = conn.getresponse()
    dic = json.loads(res.read().decode("utf-8"))
    return format_search_results(dic)
class GraphState(TypedDict):
    question: str
    generation: str
    datasource: Literal["web_search", "rag", "llm"]
def route_question_node(state: GraphState,classifier) -> dict:
    question = state["question"]
    datasource = router(question,classifier)
    return {"datasource": datasource}
# Node 2: Call Web Search (Uses user's web_search + LLM synthesis)
def call_web_search_node(state: GraphState,SERPER_API_KEY,llm) -> dict:
    question = state["question"]
    search_context = web_search(question, SERPER_API_KEY)
    synthesis_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an AI assistant. Based *only* on the provided web search context, answer the user's question concisely. If the context doesn't contain the answer or there was an error, state that clearly."),
            ("human", "Web Search Context:\n{context}\n\nQuestion:\n{question}"),
        ]
    )
    synthesis_chain = synthesis_prompt | llm | StrOutputParser()
    final_answer = synthesis_chain.invoke({"context": search_context, "question": question})
    return {"generation": final_answer}
# Node 3: Perform RAG (Uses ChromaDB retriever with Google Embeddings + LLM synthesis)
def perform_rag_node(state: GraphState, vectorstore, llm) -> dict:
    results = vectorstore.get()
    # Find the IDs of documents with empty page_content
    empty_doc_ids = []
    for i, doc_content in enumerate(results['documents']):
        if doc_content.strip() == '':
            empty_doc_ids.append(results['ids'][i])

    # Delete the empty documents
    if empty_doc_ids:
        vectorstore.delete(ids=empty_doc_ids)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    question = state["question"]
    retrieved_docs = retriever.get_relevant_documents(question)
    if not retrieved_docs:
        rag_answer = "I couldn't find relevant information in the internal crypto documents for your query."
        return {"generation": rag_answer}
    formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    rag_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an AI assistant specialized in crypto and decentralised finance topics based on internal documents. Answer the user's question based *only* on the provided document context. If the context doesn't contain the answer, state that clearly based *only* on the provided documents."),
                ("human", "Document Context:\n{context}\n\nQuestion:\n{question}"),
            ]
        )

    rag_chain = rag_prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"context": formatted_context, "question": question})
    return {"generation": generation}
# Node 4: Generate Direct LLM Response (Same as before)
def generate_direct_llm_node(state: GraphState, llm) -> dict:
    question = state["question"]
    direct_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI assistant. Answer the user's question directly."),
            ("human", "{question}"),
        ]
    )
    direct_chain = direct_prompt | llm | StrOutputParser()
    generation = direct_chain.invoke({"question": question})
    return {"generation": generation}
# --- Define Conditional Edge Logic (Same as before) ---
def decide_next_node(state: GraphState) -> str:
    datasource = state['datasource']
    if datasource == "web_search":
        return "call_web_search_node"
    elif datasource == "rag":
        return "perform_rag_node"
    elif datasource == "llm":
        return "generate_direct_llm_node"
    else:
        logging.warning(f"Unexpected datasource '{datasource}' encountered in conditional edge. Defaulting to LLM.")
        return "generate_direct_llm_node"
