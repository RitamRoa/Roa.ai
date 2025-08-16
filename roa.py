import os
import time
from typing import Dict, Any
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from langgraph.graph import StateGraph, END

os.environ["GOOGLE_API_KEY"] = "AIzaSyBys6zKt9RtgAOEjYrLvq6CjxAkqLGxSzQ"
OPENWEATHERMAP_API_KEY = "0d72d2c544944b4f8baeb2889908a64c" 
GNEWS_API_KEY = "849de1cac921edd85c35a5c6c4c089f5" 
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)

class ExpertSystemState(BaseModel):
    user_input: str
    category: str = None
    question_type: str = None
    response: str = None
def log_event(step: str, message: str):
    print(f"[{step}] {message}")
def super_agent(state: ExpertSystemState):
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
    """Handles weather-related queries via LangGraph - this is the AI-routed path."""
    log_event("weather", "AI-routed weather. Attempting to fetch real-time weather data.")
    city = "Bengaluru"
    return _fetch_weather_data(city) 

def news(state: ExpertSystemState) -> Dict[str, Any]:
    """Handles news-related queries via LangGraph - this is the AI-routed path."""
    log_event("news", "AI-routed news. Attempting to fetch real-time news headlines.")
    query_topic = "latest headlines" 
    return _fetch_news_data(query_topic) 

def joke(state: ExpertSystemState) -> Dict[str, Any]:
    """Handles joke-related queries by generating a joke using Gemini."""
    log_event("joke", "Generating a joke using Gemini.")
    response_text = llm.invoke(f"Tell a short, family-friendly joke based on the query: '{state.user_input}'").content.strip()
    log_event("joke", f"Generated joke response: {response_text}")
    return {"response": response_text}

def others(state: ExpertSystemState) -> Dict[str, Any]:
    """Handles general information queries by generating a response with the LLM."""
    log_event("Others Service", "Handling a general information query.")
    response = llm.invoke(f"Provide a sharp, short answer to the following question: {state.user_input}").content.strip()
    return {"response": response}

def route_decision(state: ExpertSystemState):
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
app_langgraph = workflow.compile()
def _fetch_weather_data(city: str) -> Dict[str, Any]:
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}q={city}&appid={OPENWEATHERMAP_API_KEY}&units=metric"

    try:
        response = requests.get(complete_url)
        data = response.json()

        if data.get("cod") == 200:
            main_data = data.get("main", {})
            weather_data = data.get("weather", [])
            
            temperature = main_data.get("temp")
            feels_like = main_data.get("feels_like")
            description = weather_data[0].get("description") if weather_data else "not available"
            humidity = main_data.get("humidity")
            wind_speed = data.get("wind", {}).get("speed")

            response_text = (
                f"The weather in {city} is {description}. "
                f"Temperature: {temperature}°C (feels like {feels_like}°C). "
                f"Humidity: {humidity}%. Wind speed: {wind_speed} m/s."
            )
            log_event("_fetch_weather_data", f"Successfully fetched weather for {city}.")
        else:
            error_message = data.get("message", "Unknown error fetching weather.")
            response_text = f"Could not fetch weather for {city}. Error: {error_message}. (Is OpenWeatherMap API key valid or quota exceeded?)"
            log_event("_fetch_weather_data", f"Failed to fetch weather: {error_message}")
    except requests.exceptions.RequestException as e:
        response_text = f"Network error fetching weather: {e}. Please check your internet connection or API endpoint."
        log_event("_fetch_weather_data", f"Network error: {e}")
    except Exception as e:
        response_text = f"An unexpected error occurred while processing weather data: {e}"
        log_event("_fetch_weather_data", f"Unexpected error: {e}")

    return {"response": response_text}

def _fetch_news_data(topic: str) -> Dict[str, Any]:
    base_url = "https://gnews.io/api/v4/top-headlines?"
    complete_url = f"{base_url}lang=en&country=in&max=5&token={GNEWS_API_KEY}" 

    try:
        response = requests.get(complete_url)
        data = response.json()

        if data.get("totalArticles") and data.get("articles"): 
            articles = data["articles"]
            news_headlines = "Here are some top headlines:\n"
            for i, article in enumerate(articles[:5]):
                title = article.get("title", "No title")
                source = article.get("source", {}).get("name", "Unknown source")
                news_headlines += f"{i+1}. {title} (Source: {source})\n"
            response_text = news_headlines
            log_event("_fetch_news_data", f"Successfully fetched news for topic: {topic}.")
        else:
            response_text = "No top headlines found at this moment. Please check your GNews API key, ensure it's active, or try again later. It might also be a quota issue. 😔"
            log_event("_fetch_news_data", "GNews API returned no articles or totalArticles is zero.")
    except requests.exceptions.RequestException as e:
        response_text = f"Network error fetching news: {e}. Please check your internet connection or API endpoint."
        log_event("_fetch_news_data", f"Network error: {e}")
    except Exception as e:
        response_text = f"An unexpected error occurred while processing news data: {e}"
        log_event("_fetch_news_data", f"Unexpected error: {e}")

    return {"response": response_text}


app_flask = Flask(__name__)
CORS(app_flask) 
@app_flask.route('/')
def serve_index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'index.html')
@app_flask.route('/ask', methods=['POST'])
def ask_expert_system():
    user_query = request.json.get('query')
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    if user_query.lower() in ["quit", "exit"]:
        return jsonify({"response": "Session ended. Please restart for new queries."})

    start_time = time.time()
    try:
        final_state = app_langgraph.invoke({"user_input": user_query})
        response_content = final_state['response']
        end_time = time.time()
        return jsonify({
            "response": response_content,
            "time_taken": f"{end_time - start_time:.2f} seconds"
        })
    except Exception as e:
        print(f"Error during AI processing: {e}")
        return jsonify({"error": str(e)}), 500

@app_flask.route('/weather_bengaluru', methods=['GET'])
def get_bengaluru_weather():
    """Directly fetches weather for Bengaluru without LangGraph routing."""
    log_event("API Direct", "Received request for Bengaluru weather.")
    start_time = time.time()
    result = _fetch_weather_data("Bengaluru")
    end_time = time.time()
    return jsonify({
        "response": result["response"],
        "time_taken": f"{end_time - start_time:.2f} seconds"
    })

@app_flask.route('/news_headlines', methods=['GET'])
def get_news_headlines():
    """Directly fetches top headlines without LangGraph routing."""
    log_event("API Direct", "Received request for top headlines.")
    start_time = time.time()
    result = _fetch_news_data("latest headlines") 
    end_time = time.time()
    return jsonify({
        "response": result["response"],
        "time_taken": f"{end_time - start_time:.2f} seconds"
    })

if __name__ == '__main__':
    print("Starting Flask API on http://127.0.0.1:5000")
    app_flask.run(host='0.0.0.0', port=5000)
