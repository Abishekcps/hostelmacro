import streamlit as st
import sentry_sdk
from duckduckgo_search import DDGS
import ollama

# 1. Initialize Sentry (Paste your Sentry DSN link here)
sentry_sdk.init(
    dsn="https://2e2a217b94ae89cc05f6b53a7683cc37@o4512202172268544.ingest.de.sentry.io/4512202180264016"
, 
    enable_tracing=True,
    traces_sample_rate=1.0,
)

def search_nutrition(query):
    """Fetches real-time nutritional context via DuckDuckGo without an API key."""
    with sentry_sdk.start_transaction(op="tool", name="DuckDuckGo_Search"):
        try:
            # We use DuckDuckGo's keyless search package
            results = DDGS().text(f"{query} nutritional facts calories protein", max_results=3)
            snippets = [item.get("body", "") for item in results]
            return " ".join(snippets)
        except Exception as e:
            sentry_sdk.capture_exception(e)
            return "Could not fetch live data."

def get_gemma_recommendation(menu, goal, web_context):
    """Runs local inference using Gemma 2 via Ollama."""
    with sentry_sdk.start_transaction(op="agent", name="Gemma2_Local_Inference"):
        prompt = f"""
        You are a nutritional assistant for a student trying to hit fitness goals.
        Today's Mess Menu: {menu}
        User's Goal: {goal}
        Live Web Nutritional Context: {web_context}
        
        Provide a short, structured meal plan picking the best items from the menu to hit their goal. Keep it under 100 words.
        """
        response = ollama.chat(
            model='gemma2:2b',
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']

# --- Streamlit UI ---
st.set_page_config(page_title="HostelMacro AI", page_icon="🍛")
st.title("🍛 HostelMacro AI")
st.write("Offline-first macro tracking for the hostel mess.")

menu_input = st.text_area("What's on the menu today?", "Dal fry, paneer butter masala, 3 rotis, rice")
goal_input = st.text_input("What is your goal?", "I need 40g of protein and want to stay in a calorie deficit.")

if st.button("Generate Plate"):
    with st.spinner("Searching DuckDuckGo & generating locally..."):
        context = search_nutrition(menu_input)
        plan = get_gemma_recommendation(menu_input, goal_input, context)
        
        st.success("Meal Plan Generated!")
        st.markdown(f"> **Agent Output:**\n\n{plan}")