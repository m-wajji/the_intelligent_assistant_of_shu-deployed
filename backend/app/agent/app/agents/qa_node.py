import re
from typing import List, Dict, Tuple
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from app.agent.app.core.state import State
from langgraph.graph import END
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

# Initialize embeddings model
embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize multiple vectorstores
VECTORSTORES = {
    "academic_programs": Chroma(
        persist_directory="app/shu_partitioned_db/chroma_dbs/shu_academic_programs_db/",
        embedding_function=embedding_function,
        collection_name="shu_academic_programs_db" 
    ),
    "admissions_services": Chroma(
        persist_directory="app/shu_partitioned_db/chroma_dbs/shu_admissions_services_db/",
        embedding_function=embedding_function,
        collection_name="shu_admissions_services_db"
    ),
    "events_news": Chroma(
        persist_directory="app/shu_partitioned_db/chroma_dbs/shu_events_news_db/",
        embedding_function=embedding_function,
        collection_name="shu_events_news_db"  
    ),
    "financial_operational": Chroma(
        persist_directory="app/shu_partitioned_db/chroma_dbs/shu_financial_operational_db/",
        embedding_function=embedding_function,
        collection_name="shu_financial_operational_db" 
    ),
    "institutional_governance": Chroma(
        persist_directory="app/shu_partitioned_db/chroma_dbs/shu_institutional_governance_db/",
        embedding_function=embedding_function,
        collection_name="shu_institutional_governance_db"
    ),
    "research_innovation": Chroma(
        persist_directory="app/shu_partitioned_db/chroma_dbs/shu_research_innovation_db/",
        embedding_function=embedding_function,
        collection_name="shu_research_innovation_db"
    ),
    "student_life_services": Chroma(
        persist_directory="app/shu_partitioned_db/chroma_dbs/shu_student_life_services_db/",
        embedding_function=embedding_function,
        collection_name="shu_student_life_services_db" 
    ),
}

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
classifier_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Query classification keywords and patterns
QUERY_CLASSIFICATIONS = {
    "academic_programs": [
        "degree", "program", "course", "curriculum", "major", "minor", "bachelor", 
        "master", "phd", "doctorate", "diploma", "certificate", "syllabus", 
        "subjects", "computer science", "engineering", "business", "arts", 
        "science", "faculty", "department", "academic", "study", "graduation","dean","chairperson", "hod"
    ],
    "admissions_services": [
        "admission", "apply", "application", "entry", "requirement", "eligibility", 
        "entrance", "test", "exam", "score", "gpa", "merit", "deadline", 
        "fee structure", "scholarship", "financial aid", "enroll", "registration"
    ],
    "events_news": [
        "event", "news", "announcement", "seminar", "workshop", "conference", 
        "competition", "ceremony", "celebration", "activity", "happening", 
        "upcoming", "recent", "latest", "update", "competition", "fbs"
    ],
    "financial_operational": [
        "fee", "tuition", "cost", "payment", "financial", "budget", "expense", 
        "installment", "refund", "billing", "accounts", "finance", "operational",
        "facilities", "infrastructure", "campus", "library", "lab"
    ],
    "institutional_governance": [
        "administration", "management", "governance", "policy", "procedure", 
        "leadership", "board", "committee", "director", "chancellor", "vice chancellor",
        "dean", "head", "organizational", "structure", "hierarchy"
    ],
    "research_innovation": [
        "research", "innovation", "project", "publication", "thesis", "dissertation", 
        "journal", "conference paper", "patent", "collaboration", "lab", 
        "investigation", "study", "experiment", "development"
    ],
    "student_life_services": [
        "student life", "hostel", "accommodation", "transport", "cafeteria", 
        "dining", "sports", "recreation", "club", "society", "health", 
        "counseling", "support", "services", "extracurricular", "activities"
    ]
}

def classify_query(query: str) -> List[str]:
    """
    Classify the query to determine which databases to search.
    Returns a list of relevant database categories.
    """
    query_lower = query.lower()
    relevant_dbs = []
    
    # Score each category based on keyword matches
    category_scores = {}
    for category, keywords in QUERY_CLASSIFICATIONS.items():
        score = sum(1 for keyword in keywords if keyword in query_lower)
        if score > 0:
            category_scores[category] = score
    
    # If we have keyword matches, return top scoring categories
    if category_scores:
        # Sort by score and return top categories (max 3 to avoid too much noise)
        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        relevant_dbs = [cat for cat, score in sorted_categories[:3]]
    
    # If no clear matches, use LLM for classification
    if not relevant_dbs:
        classification_prompt = f"""
        Classify the following query about Salim Habib University into one or more of these categories:
        
        1. academic_programs - Degrees, courses, curriculum, departments, faculty
        2. admissions_services - Applications, requirements, eligibility, entrance tests
        3. events_news - Events, announcements, news, seminars, workshops
        4. financial_operational - Fees, costs, facilities, infrastructure, operations
        5. institutional_governance - Administration, leadership, policies, management
        6. research_innovation - Research projects, publications, innovation, labs
        7. student_life_services - Student services, hostel, sports, clubs, support
        
        Query: "{query}"
        
        Respond with only the category name(s) separated by commas. Choose the most relevant 1-2 categories.
        """
        
        try:
            response = classifier_llm.invoke([HumanMessage(content=classification_prompt)])
            categories = [cat.strip() for cat in response.content.split(',')]
            relevant_dbs = [cat for cat in categories if cat in VECTORSTORES.keys()]
        except Exception:
            # If classification fails
            relevant_dbs = ["academic_programs", "admissions_services"]
    
    # If still no matches, search the most general categories
    if not relevant_dbs:
        relevant_dbs = ["academic_programs", "admissions_services"]
    
    return relevant_dbs

def search_multiple_docs(query: str, db_categories: List[str], k: int = 3) -> Tuple[str, List[str]]:
    """
    Search multiple databases and return combined results with source information.
    """
    all_docs = []
    sources_used = []
    
    print(f"\nSearching databases: {db_categories}")
    
    for category in db_categories:
        if category in VECTORSTORES:
            try:
                docs = VECTORSTORES[category].similarity_search(query, k=k)
                if docs:
                    sources_used.append(category)
                    print(f"Found {len(docs)} documents in {category}")
                    
                    for i, doc in enumerate(docs):
                        # Print document metadata for monitoring
                        print(f"Doc {i+1} metadata: {doc.metadata}")
                        
                        # Add source category to the document content
                        doc_content = f"[Source: {category.replace('_', ' ').title()}]\n{doc.page_content}"
                        all_docs.append(doc_content)
            except Exception as e:
                print(f"Error searching {category}: {e}")
                continue
    
    combined_docs = "\n\n---\n\n".join(all_docs)
    print(f"Total documents retrieved: {len(all_docs)}")
    
    return combined_docs, sources_used

# Define the enhanced search tool
def enhanced_search_docs(query: str) -> str:
    """Enhanced search function that uses multiple databases."""
    # Expand SHU abbreviations
    expanded_query = re.sub(r'\bshu\b', 'Salim Habib University', query, flags=re.IGNORECASE)
    
    # Classify query and search relevant databases
    relevant_dbs = classify_query(expanded_query)
    print(f"Query classified to: {relevant_dbs}")
    
    docs, sources = search_multiple_docs(expanded_query, relevant_dbs, k=5)
    
    return docs

search_tool = Tool(
    name="enhanced_search_docs",
    func=enhanced_search_docs,
    description="Search multiple specialized databases about Salim Habib University based on query context.",
)

def qa_node(state: State) -> Command[str]:
    """
    Enhanced QA node that uses multiple specialized databases and provides contextual responses.
    """
    user_query = state["input"]
    
    print(f"\n{'='*60}")
    print(f"Processing query: {user_query}")
    print(f"{'='*60}")
    
    # Get conversation history
    history_msgs = memory.load_memory_variables(
        {"chat_history": state.get("messages", [])}
    )["chat_history"]
    
    # Search for relevant information
    docs = enhanced_search_docs(user_query)
    
    if not docs.strip():
        ai_reply = (
            "I couldn't find specific information about your query. "
            "Please visit Salim Habib University's official website or contact the relevant "
            "department for the most accurate and up-to-date information."
        )
    else:
        system_prompt = system_prompt = """You are an intelligent assistant for Salim Habib University (SHU). Your role is to provide accurate, helpful, and comprehensive information about the university based on the provided documents.

        Guidelines for your responses:
        1. Answer directly and concisely while being comprehensive
        2. Use information ONLY from the provided documents
        3. If the documents don't contain enough information for a complete answer, provide what information is available and then simply state: "For more detailed information, please visit the university's official website."
        4. Maintain a helpful and professional tone
        5. When multiple sources provide information, synthesize them coherently
        6. If asked about specific procedures or requirements, provide step-by-step information when available
        7. For numerical information (fees, dates, scores), be precise and cite the source section
        8. If the query is completely outside the scope of the provided documents, respond with: "This information is not available in my current knowledge. Please visit the university's official website for this information."

        Remember: You represent Salim Habib University, so maintain professionalism while being approachable and helpful. Keep redirections brief and direct."""

        messages = [
            SystemMessage(content=system_prompt),
            *history_msgs,
            HumanMessage(
                content=f"""User Query: {user_query}
                Relevant Information from University Database:
                {docs}
                Please provide a comprehensive answer based on the information above."""
            )
        ]
        
        try:
            response = llm.invoke(messages)
            ai_reply = response.content
        except Exception as e:
            print(f"LLM Error: {e}")
            ai_reply = (
                "I'm experiencing technical difficulties processing your query. "
                "Please try again or contact Salim Habib University directly for assistance."
            )

    memory.save_context({"input": user_query}, {"output": ai_reply})
    
    print(f"Query processed successfully\n{'='*60}")
    
    # Update state with the response
    return Command(
        update={
            "messages": state.get("messages", [])
            + [AIMessage(content=ai_reply, name="qa_agent")]
        },
        goto=END,
    )
