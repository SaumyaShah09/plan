# Add these three lines to override the default sqlite3 library
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from crewai import Agent, Task, Crew
from crewai.llm import LLM
import streamlit as st
from dotenv import load_dotenv
from serpapi import GoogleSearch
import os
from datetime import date, timedelta

# --- API Key and LLM Configuration ---
# It's better to ask for keys in the UI for deployed apps
# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

st.set_page_config(page_title="AI Trip Designer")
st.title("📜 Multi-Agent Trip Planner")

# --- API Key Input in Sidebar ---
st.sidebar.header("API Configuration")
groq_api_key = st.sidebar.text_input(
    "Enter your GROQ API Key", type="password", help="Get your key from https://console.groq.com/keys"
)
serpapi_api_key = st.sidebar.text_input(
    "Enter your SerpAPI API Key", type="password", help="Get your key from https://serpapi.com/dashboard"
)

# Initialize LLM only if the key is provided
llm = None
if groq_api_key:
    llm = LLM(
        model="llama3-70b-8192",
        api_key=groq_api_key,
        base_url="https://api.groq.com/openai/v1"
    )

# --- Session State Management ---
if "chat_stage" not in st.session_state:
    st.session_state.chat_stage = "collecting"
    st.session_state.user_data = {}
    st.session_state.last_input = None
    st.session_state.itinerary_text = ""
    st.session_state.finalized = False
    st.session_state.flight_info = ""
    st.session_state.hotel_info = ""

# --- Helper Functions for API Calls ---
def get_flight_info(departure, arrival, days):
    iata_mapping = {
        "ahmedabad": "AMD", "goa": "GOI", "delhi": "DEL", "mumbai": "BOM",
        "bangalore": "BLR", "chennai": "MAA", "kolkata": "CCU", "hyderabad": "HYD",
        "jaipur": "JAI", "kochi": "COK", "paris": "CDG", "london": "LHR",
        "new york": "JFK", "los angeles": "LAX", "tokyo": "NRT", "dubai": "DXB",
        "singapore": "SIN", "sydney": "SYD", "toronto": "YYZ", "frankfurt": "FRA",
        "hong kong": "HKG", "amsterdam": "AMS", "bangkok": "BKK", "shanghai": "PVG",
        "beijing": "PEK", "seoul": "ICN", "doha": "DOH", "zurich": "ZRH", "kuala lumpur": "KUL"
    }

    departure_code = iata_mapping.get(departure.lower().strip())
    arrival_code = iata_mapping.get(arrival.lower().strip())

    if not departure_code or not arrival_code:
        return f"❌ Could not find airport code for {departure} or {arrival}."

    depart_date = (date.today() + timedelta(days=30)).isoformat()
    return_date = (date.today() + timedelta(days=30 + days)).isoformat()

    params = {
        "engine": "google_flights",
        "departure_id": departure_code,
        "arrival_id": arrival_code,
        "outbound_date": depart_date,
        "return_date": return_date,
        "currency": "INR",
        "hl": "en",
        "gl": "in",
        "api_key": serpapi_api_key
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        
        if "error" in results:
            return f"❌ SerpAPI Error: {results['error']}"
            
        flights = results.get("best_flights", []) or results.get("other_flights", [])

        if not flights:
            return "❌ No flights found for the specified route and dates."

        lines = ["**✈️ Flight Options:**"]
        for f in flights[:3]:
            price = f.get("price", "N/A")
            carrier = "/".join(list(set(seg.get("airline", "-") for seg in f.get("flights", []))))
            lines.append(f"- **{carrier}**: ₹{price:,}" if isinstance(price, int) else f"- **{carrier}**: ₹{price}")
        return "\n".join(lines)

    except Exception as e:
        return f"❌ Exception occurred while fetching flights: {e}"


def get_hotel_info(city, budget):
    check_in = (date.today() + timedelta(days=30)).isoformat()
    check_out = (date.today() + timedelta(days=34)).isoformat()

    params = {
        "engine": "google_hotels",
        "q": f"hotels in {city}",
        "currency": "INR",
        "check_in_date": check_in,
        "check_out_date": check_out,
        "adults": "2",
        "hl": "en",
        "gl": "in",
        "api_key": serpapi_api_key
    }
    
    try:
        search = GoogleSearch(params)
        results = search.get_dict()

        if "error" in results:
            return f"❌ SerpAPI Error: {results['error']}"

        props = results.get("properties", [])
        if not props:
            return "❌ No hotels found via SerpAPI."

        price_caps = {"low-range": 4000, "mid-range": 9000, "luxury": 90000}
        cap = price_caps.get(budget, 9000)
        
        shortlist = ["**🏨 Hotel Options:**"]
        for h in props:
            price = h.get("rate_per_night", {}).get("extracted_lowest")
            if price and price <= cap:
                shortlist.append(f"- **{h['name']}**: ₹{int(price):,}/night")
            if len(shortlist) == 4: # 1 for header, 3 for hotels
                break
        
        return "\n".join(shortlist) if len(shortlist) > 1 else "❌ No hotels matched the chosen budget."

    except Exception as e:
        return f"❌ Exception occurred while fetching hotels: {e}"


def generate_full_itinerary(prefs):
    flights_txt = get_flight_info(prefs['departure'], prefs['destination'], prefs['days'])
    hotels_txt = get_hotel_info(prefs['destination'], prefs['budget'])

    itinerary_planner = Agent(
        role="Expert Travel Planner",
        goal=f"Craft a detailed, day-by-day itinerary for a {prefs['days']}-day trip to {prefs['destination']}",
        backstory="An expert in crafting luxurious and culturally rich travel experiences.",
        llm=llm,
        verbose=True
    )

    itinerary_task = Task(
      description=f"""
        Create a detailed, day-by-day itinerary for a {prefs['days']}-day luxury trip to {prefs['destination']} for a client departing from {prefs['departure']}.
        The client has a {prefs['budget']} budget.

        Incorporate the following real-time information for flights and hotels into your plan:
        - **Flight Options:** {flights_txt}
        - **Hotel Options:** {hotels_txt}

        **Instructions:**
        1.  Start with a brief, engaging summary of the trip.
        2.  Recommend one flight and one hotel from the options provided, explaining why it's a good fit.
        3.  Create a detailed plan for each day, from morning to evening. Suggest specific activities, restaurants, and sights.
        4.  Ensure the activities are logical for the location and fit a luxury travel style.
        5.  The final output must be a complete, well-formatted markdown document. Do not include any preliminary thoughts or notes.
      """,
      expected_output="A complete, well-formatted markdown itinerary, including a trip summary, flight/hotel recommendations, and a detailed daily plan.",
      agent=itinerary_planner
    )

    crew = Crew(agents=[itinerary_planner], tasks=[itinerary_task])
    result = crew.kickoff()
    return flights_txt, hotels_txt, result

# --- Main App Logic ---
if not llm or not serpapi_api_key:
    st.info("Please enter your API keys in the sidebar to start planning your trip.")
    st.stop()

# Display chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def ask_question(question):
    st.session_state.messages.append({"role": "assistant", "content": question})
    with st.chat_message("assistant"):
        st.markdown(question)

# Handle user input and conversation flow
if st.session_state.chat_stage == "collecting":
    if not st.session_state.user_data.get("destination"):
        ask_question("Where would you like to travel?")
    elif not st.session_state.user_data.get("days"):
        ask_question("How many days will your trip be?")
    elif not st.session_state.user_data.get("budget"):
        ask_question("What's your budget? (low-range, mid-range, luxury)")
    elif not st.session_state.user_data.get("departure"):
        ask_question("Which city will you be departing from?")

user_input = st.chat_input("Your response...")

if user_input:
    st.session_state.last_input = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": st.session_state.last_input})
    with st.chat_message("user"):
        st.markdown(st.session_state.last_input)

    if st.session_state.chat_stage == "collecting":
        if not st.session_state.user_data.get("destination"):
            st.session_state.user_data["destination"] = st.session_state.last_input
            st.session_state.last_input = None
            st.rerun()
        elif not st.session_state.user_data.get("days"):
            try:
                st.session_state.user_data["days"] = int(st.session_state.last_input)
                st.session_state.last_input = None
                st.rerun()
            except ValueError:
                ask_question("Please enter a valid number for the number of days.")
                st.session_state.last_input = None
        elif not st.session_state.user_data.get("budget"):
            st.session_state.user_data["budget"] = st.session_state.last_input.lower().strip()
            st.session_state.last_input = None
            st.rerun()
        elif not st.session_state.user_data.get("departure"):
            st.session_state.user_data["departure"] = st.session_state.last_input
            st.session_state.chat_stage = "planning"
            st.session_state.last_input = None
            st.rerun()

# --- Itinerary Generation ---
if st.session_state.chat_stage == "planning" and not st.session_state.itinerary_text:
    with st.spinner("Agents are crafting your perfect itinerary... This may take a moment."):
        flights, hotels, itinerary = generate_full_itinerary(st.session_state.user_data)
        st.session_state.flight_info = flights
        st.session_state.hotel_info = hotels
        st.session_state.itinerary_text = itinerary
        st.session_state.chat_stage = "reviewing"
        st.rerun()

# --- Display Final Itinerary ---
if st.session_state.chat_stage == "reviewing":
    final_response = f"{st.session_state.flight_info}\n\n{st.session_state.hotel_info}\n\n**📋 Your Custom Itinerary:**\n\n{st.session_state.itinerary_text}"
    st.session_state.messages.append({"role": "assistant", "content": final_response})
    with st.chat_message("assistant"):
        st.markdown(final_response)
    st.session_state.chat_stage = "done" # Prevent re-running
