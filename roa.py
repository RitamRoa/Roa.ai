import os
import time
from typing import Annotated, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from langgraph.graph import StateGraph, END

os.environ["GOOGLE_API_KEY"] = "AIzaSyBys6zKt9RtgAOEjYrLvq6CjxAkqLGxSzQ"
print("API key set")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)

class ExpertSystemState(BaseModel):
    user_input: str
    category: str = None
    question_type: str = None
    response: str = None

def log_event(step: str, message: str):
    """Log an event with message and step."""
    print(f"[{step}] {message}")

def super_agent(state: ExpertSystemState):
    """
    Analyze user's query and determine the appropriate service.

    Args:
        state (ExpertSystemState): The current state of the graph.

    Returns:
        Dict[str, Any]: The response from the appropriate service.
    """
    input_data = state.user_input
    prompt_template = f"""
    Categorize the query into the following categories:
    1. weather
    2. news
    3. joke
    4. others
    Query: {input_data}
    Category:
    """
    log_event("super_agent", f"Processing query: '{input_data}'")
    categorization_response = llm.invoke(prompt_template).content.strip()

    if "weather" in categorization_response.lower():
        category = "weather"
    elif "news" in categorization_response.lower():
        category = "news"
    elif "joke" in categorization_response.lower():
        category = "joke"
    else:
        category = "others"
    log_event("super_agent", f"Categorized as: {category}")
    return {"category": category}

def weather(state: ExpertSystemState) -> Dict[str, Any]:
    """Handles weather-related queries."""
    log_event("weather", "Handling weather query")
    response = "The weather is sunny with a high of 25°C."
    return {"response": response}

def news(state: ExpertSystemState) -> Dict[str, Any]:
    """Handles news-related queries."""
    log_event("News Service", "Providing a news-related response.")
    response = "The latest headlines include a breakthrough in renewable energy and a new study on climate change. 🗞️"
    return {"response": response}

def joke(state: ExpertSystemState) -> Dict[str, Any]:
    """Handles joke-related queries."""
    log_event("Joke Service", "Providing a joke.")
    response = "Why don't scientists trust atoms? Because they make up everything! 😂"
    return {"response": response}

def others(state: ExpertSystemState) -> Dict[str, Any]:
    """Handles general information queries by generating a response with the LLM."""
    log_event("Others Service", "Handling a general information query.")

    response = llm.invoke(f"Provide a sharp, short answer to the following question: {state.user_input}").content.strip()

    return {"response": response}

def route_decision(state: ExpertSystemState):
    """Route the query to the appropriate service based on the category."""
    log_event("Route Decision", f"Routing based on category: {state.category}")
    return state.category

workflow = StateGraph(ExpertSystemState)
workflow.add_node("super_agent", super_agent)
workflow.add_node("weather", weather)
workflow.add_node("news", news)
workflow.add_node("joke", joke)
workflow.add_node("others", others)

workflow.set_entry_point("super_agent")

workflow.add_conditional_edges(
    "super_agent",
    route_decision,
    {
        "weather": "weather",
        "news": "news",
        "joke": "joke",
        "others": "others",
    },
)


workflow.add_edge("weather", END)
workflow.add_edge("news", END)
workflow.add_edge("joke", END)
workflow.add_edge("others", END)

app = workflow.compile()

if __name__ == "__main__":
    print("I am Roa, ask question or type 'quit' or 'exit' to end the session.")
    while True:
    
        user_input = input("You: ").strip()
        
        if user_input.lower() in ["quit", "exit"]:
            print("Dont waste my time")
            break

        print(f"\nUser Query: {user_input}")
        start_time = time.time()

        final_state = app.invoke({"user_input": user_input})

        end_time = time.time()
        
        print(f"Final Response: {final_state['response']}")
        print(f"Total time taken: {end_time - start_time:.2f} seconds")
        print("-" * 50)