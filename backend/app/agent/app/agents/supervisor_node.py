from typing_extensions import TypedDict
from langgraph.types import Command
from langgraph.graph import END
from app.agent.app.core.state import State
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
# System prompt for LLM
# system_prompt = (
#     "You are a supervisor agent managing different workers: ['qa_agent', 'pdf_agent'].\n"
#     "Given a user message, select which worker should respond:\n"
#     "- Use `qa_agent` for general questions about SHU, like 'where is SHU?' or 'who is the chancellor?'\n"
#     "- Use `pdf_agent` if the user asks for documents, like 'give me the prospectus PDF', 'abstract book', or 'upload a new PDF'.\n"
#     "If the user’s query is fully answered, return 'FINISH'."

# )

system_prompt = (
    "You are a routing (“supervisor”) agent for SHU Assistant."
    "There are exactly two worker agents:"
    "   • pdf_agent — handles any request for documents or PDF forms"
    "   • qa_agent  — handles all other university questions (location, admissions, contacts, etc.)\n"
    "When given a user message, pick exactly one of those agents."
    "Respond with a single JSON object and nothing else, for example:\n"
    "{'next': 'qa_agent'}\n"
    "or\n"
    "{'next': 'pdf_agent'}\n"
    "Do NOT output any other keys, punctuation, or commentary."
    "Always choose pdf_agent if and only if the user is explicitly requesting a PDF, document, form, brochure, syllabus, or similar. "
    "For every other query, choose qa_agent.".strip()
)


# The list of actual worker‐node names in your graph
MEMBERS = [
    "supervisor",  # your dispatcher
    "pdf_agent",  # handles PDF requests
    "qa_agent",  # handles QA requests
]
supervisor_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
# The set of all routing options, including the “FINISH” sentinel
OPTIONS = ["pdf_agent", "qa_agent"]


# Define the structured output format
class Router(TypedDict):
    next: str


def supervisor_node(state: State) -> Command[str]:
    """
    Supervisor node that routes input to the appropriate agent.
    """
    messages = [{"role": "system", "content": system_prompt}] + state.get(
        "messages", []
    )

    # LLM picks the next agent
    response = supervisor_llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    print("Supervisor intent (raw):", goto)

    # Validate and fallback if invalid
    if goto not in OPTIONS:
        print(f"[WARN] Invalid intent '{goto}' — defaulting to 'qa_agent'")
        goto = "qa_agent"  # You can change this default if needed

    # Map FINISH to END sentinel
    if goto == "FINISH":
        goto = END

    # Return routing command
    return Command(goto=goto, update={"next": goto})
