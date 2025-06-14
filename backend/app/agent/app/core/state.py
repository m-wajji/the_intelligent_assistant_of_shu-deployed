from langgraph.graph import MessagesState

class State(MessagesState):
    next: str
    input: str   
