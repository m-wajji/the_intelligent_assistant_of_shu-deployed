from langgraph.graph import StateGraph ,END
from app.agent.app.core.state import State
from app.agent.app.agents.supervisor_node import supervisor_node
from app.agent.app.agents.pdf_node import pdf_node
from app.agent.app.agents.qa_node import qa_node

# Build the graph with your shared State schema
builder = StateGraph(State)

#  Register each node in the workflow
builder.add_node("supervisor", supervisor_node)
builder.add_node("pdf_agent", pdf_node)
builder.add_node("qa_agent", qa_node)

# Wire the START sentinel to the supervisor node
builder.set_entry_point("supervisor")

#  Route supervisor to pdf or qa node based on intent
builder.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {
        "pdf_agent": "pdf_agent",
        "qa_agent": "qa_agent",
        END: END  # Add this route to avoid KeyError when supervisor returns END
    }
)


# Route each agent to the END sentinel
builder.add_edge("pdf_agent", END)
builder.add_edge("qa_agent", END)

# Compile the graph into an executable
graph = builder.compile()
