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

# --- Load API Keys ---
# For local development, create a .env file with your keys
# For Streamlit Cloud, set these as secrets
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# --- Initialize LLM ---
# We initialize it once and check for the key
llm = None
if GROQ_API_KEY:
    llm = LLM(
        model="llama3-70b-8192",
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="AI Trip Designer")
st.title("📜 Multi-Agent Trip Planner")

# --- Session State Management ---
if "chat_stage" not in st.session_state:
    st.session_state.chat_stage = "collecting"
    st.session_state.user_data = {}
    st.session_state.last_input = None
    st.session_state.itinerary_text = ""
    st.session_state.flight_info = ""
    st.session_state.hotel_info = ""
    st.session_state.messages = [] # To store the history

# --- Helper Functions for API Calls ---
def get_flight_info(departure, arrival, days):
    iata_mapping = {
        "ahmedabad": "AMD", "goa": "GOI", "delhi": "DEL", "mumbai": "BOM", "bangalore": "BLR", 
        "chennai": "MAA", "kolkata": "CCU", "hyderabad": "HYD", "jaipur": "JAI", "kochi": "COK", 
        "paris": "CDG", "london": "LHR", "new york": "JFK", "los angeles": "LAX", "tokyo": "NRT", 
        "dubai": "DXB", "singapore": "SIN", "sydney": "SYD", "toronto": "YYZ", "frankfurt": "FRA",
        "hong kong": "HKG", "amsterdam": "AMS", "bangkok": "BKK", "shanghai": "PVG", "beijing": "PEK",
        "seoul": "ICN", "doha": "DOH", "zurich": "ZRH", "kuala lumpur": "KUL"
    }
    departure_code = iata_mapping.get(departure.lower().strip())
    arrival_code = iata_mapping.get(arrival.lower().strip())
    if not departure_code or not arrival_code:
        return f"❌ Could not find airport code for {departure} or {arrival}."

    depart_date = (date.today() + timedelta(days=30)).isoformat()
    return_date = (date.today() + timedelta(days=30 + days)).isoformat()
    params = {
        "engine": "google_flights", "departure_id": departure_code, "arrival_id": arrival_code,
        "outbound_date": depart_date, "return_date": return_date, "currency": "INR",
        "hl": "en", "gl": "in", "api_key": SERPAPI_API_KEY
    }
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        if "error" in results: return f"❌ SerpAPI Error: {results['error']}"
        flights = results.get("best_flights", []) or results.get("other_flights", [])
        if not flights: return "❌ No flights found for the specified route and dates."
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
        "engine": "google_hotels", "q": f"hotels in {city}", "currency": "INR",
        "check_in_date": check_in, "check_out_date": check_out, "adults": "2",
        "hl": "en", "gl": "in", "api_key": SERPAPI_API_KEY
    }
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        if "error" in results: return f"❌ SerpAPI Error: {results['error']}"
        props = results.get("properties", [])
        if not props: return "❌ No hotels found via SerpAPI."
        price_caps = {"low-range": 4000, "mid-range": 9000, "luxury": 90000}
        cap = price_caps.get(budget, 9000)
        shortlist = ["**🏨 Hotel Options:**"]
        for h in props:
            price = h.get("rate_per_night", {}).get("extracted_lowest")
            if price and price <= cap:
                shortlist.append(f"- **{h['name']}**: ₹{int(price):,}/night")
            if len(shortlist) == 4: break
        return "\n".join(shortlist) if len(shortlist) > 1 else "❌ No hotels matched the chosen budget."
    except Exception as e:
        return f"❌ Exception occurred while fetching hotels: {e}"

def generate_full_itinerary(prefs):
    flights_txt = get_flight_info(prefs['departure'], prefs['destination'], prefs['days'])
    hotels_txt = get_hotel_info(prefs['destination'], prefs['budget'])
    itinerary_planner = Agent(
        role="Expert Travel Planner",
        goal=f"Craft a detailed, day-by-day itinerary for a {prefs['days']}-day trip to {prefs['destination']}",
        backstory="An expert in crafting luxurious and culturally rich travel experiences who follows instructions precisely.",
        llm=llm, verbose=True, allow_delegation=False
    )
    itinerary_task = Task(
      description=f"""
        Your single task is to generate a complete travel itinerary based on the details provided.
        You must follow the specified format exactly. Do not add any conversational text, introductions, or explanations.
        Your entire response should be only the formatted markdown itinerary.
        **Trip Details:**
        - **Destination:** {prefs['destination']}
        - **Duration:** {prefs['days']} days
        - **Departure City:** {prefs['departure']}
        - **Budget:** {prefs['budget']}
        **Available Data (for context):**
        - **Flight Options:** {flights_txt}
        - **Hotel Options:** {hotels_txt}
        **Required Output Format:**
        ### Trip Summary
        A brief, one-paragraph summary of the exciting trip ahead.
        ### Recommendations
        * **Flight:** [Recommend one flight from the options and provide a one-sentence reason for your choice.]
        * **Hotel:** [Recommend one hotel from the options and provide a one-sentence reason for your choice.]
        ### Daily Itinerary
        **Day 1:**
        * **Morning:** [Detailed activity]
        * **Afternoon:** [Detailed activity]
        * **Evening:** [Detailed activity]
        ... continue for all {prefs['days']} days ...
        **IMPORTANT:** Your response must start with '### Trip Summary' and end with the last activity of the final day.
      """,
      expected_output="A complete, well-formatted markdown itinerary that strictly follows the specified format.",
      agent=itinerary_planner
    )
    crew = Crew(agents=[itinerary_planner], tasks=[itinerary_task])
    result = crew.kickoff()
    if result and "Trip Summary" in result and "Daily Itinerary" in result:
        return flights_txt, hotels_txt, result
    else:
        return flights_txt, hotels_txt, "❌ The AI agent failed to generate a valid itinerary. Please try again."

# --- Main App Logic ---
if not llm or not SERPAPI_API_KEY:
    st.info("Please ensure GROQ_API_KEY and SERPAPI_API_KEY are set in your environment or Streamlit secrets.")
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Original UI Logic (Restored) ---
user_input = st.chat_input("Say something...")
if user_input:
    st.session_state.last_input = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": st.session_state.last_input})
    with st.chat_message("user"):
        st.markdown(st.session_state.last_input)
    st.rerun()

if st.session_state.chat_stage == "collecting":
    # Ask the next question if there's no pending user input to process
    if not st.session_state.last_input:
        if "destination" not in st.session_state.user_data:
            st.chat_message("assistant").markdown("Where would you like to travel?")
        elif "days" not in st.session_state.user_data:
            st.chat_message("assistant").markdown("How many days will your trip be?")
        elif "budget" not in st.session_state.user_data:
            st.chat_message("assistant").markdown("What's your budget? (low-range, mid-range, luxury)")
        elif "departure" not in st.session_state.user_data:
            st.chat_message("assistant").markdown("Which city will you be departing from?")

    # Process user input if it exists
    if st.session_state.last_input:
        if "destination" not in st.session_state.user_data:
            st.session_state.user_data["destination"] = st.session_state.last_input
        elif "days" not in st.session_state.user_data:
            try:
                st.session_state.user_data["days"] = int(st.session_state.last_input)
            except ValueError:
                st.chat_message("assistant").markdown("Please enter a valid number.")
        elif "budget" not in st.session_state.user_data:
            st.session_state.user_data["budget"] = st.session_state.last_input.lower().strip()
        elif "departure" not in st.session_state.user_data:
            st.session_state.user_data["departure"] = st.session_state.last_input
            st.session_state.chat_stage = "planning"
        
        st.session_state.last_input = None
        st.rerun()

# --- Itinerary Generation ---
if st.session_state.chat_stage == "planning":
    with st.spinner("Agents are crafting your perfect itinerary... This may take a moment."):
        flights, hotels, itinerary = generate_full_itinerary(st.session_state.user_data)
        st.session_state.flight_info = flights
        st.session_state.hotel_info = hotels
        st.session_state.itinerary_text = itinerary
        
        final_response = f"{flights}\n\n{hotels}\n\n**📋 Your Custom Itinerary:**\n\n{itinerary}"
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        st.session_state.chat_stage = "done"
        st.rerun()
