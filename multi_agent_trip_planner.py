# Add these three lines to override the default sqlite3 library
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('p_ysqlite3')

from crewai import Agent, Task, Crew
from crewai.llm import LLM
import streamlit as st
from dotenv import load_dotenv
from serpapi import GoogleSearch
import os
from datetime import date, timedelta

# --- Streamlit Page Configuration ---
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
    st.session_state.flight_info = ""
    st.session_state.hotel_info = ""
    st.session_state.messages = []

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
        "engine": "google_flights", "departure_id": departure_code, "arrival_id": arrival_code,
        "outbound_date": depart_date, "return_date": return_date, "currency": "INR",
        "hl": "en", "gl": "in", "api_key": serpapi_api_key
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
        "hl": "en", "gl": "in", "api_key": serpapi_api_key
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
        * **Morning:** [Detailed activity, e.g., 'Visit the Louvre Museum (pre-booked tickets recommended).']
        * **Afternoon:** [Detailed activity, e.g., 'Enjoy a picnic lunch at Champ de Mars with a view of the Eiffel Tower.']
        * **Evening:** [Detailed activity, e.g., 'Take a sunset dinner cruise on the Seine River.']

        **Day 2:**
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
        return flights_txt, hotels_txt, "❌ The AI agent failed to generate a valid itinerary. This can sometimes happen due to high traffic. Please try again."

# --- Main App Logic ---
if not llm or not serpapi_api_key:
    st.info("Please enter your API keys in the sidebar to start planning your trip.")
    st.stop()

# Display chat history and current question
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle Conversation Flow ---
if st.session_state.chat_stage == "collecting":
    if "destination" not in st.session_state.user_data:
        st.chat_message("assistant").markdown("Where would you like to travel?")
    elif "days" not in st.session_state.user_data:
        st.chat_message("assistant").markdown("How many days will your trip be?")
    elif "budget" not in st.session_state.user_data:
        st.chat_message("assistant").markdown("What's your budget? (low-range, mid-range, luxury)")
    elif "departure" not in st.session_state.user_data:
        st.chat_message("assistant").markdown("Which city will you be departing from?")

user_input = st.chat_input("Say something...")

if user_input:
    st.session_state.last_input = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": st.session_state.last_input})

    if st.session_state.chat_stage == "collecting":
        if "destination" not in st.session_state.user_data:
            st.session_state.user_data["destination"] = st.session_state.last_input
        elif "days" not in st.session_state.user_data:
            try:
                st.session_state.user_data["days"] = int(st.session_state.last_input)
            except ValueError:
                st.session_state.messages.append({"role": "assistant", "content": "Please enter a valid number."})
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
        
        # Display the final result
        final_response = f"{flights}\n\n{hotels}\n\n**📋 Your Custom Itinerary:**\n\n{itinerary}"
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        st.session_state.chat_stage = "done"
        st.rerun()
