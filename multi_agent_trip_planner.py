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

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

llm = LLM(
    model="llama3-70b-8192",
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

st.set_page_config(page_title="AI Trip Designer ")
st.title("📜 Multi-Agent Trip Planner")

if "chat_stage" not in st.session_state:
    st.session_state.chat_stage = "collecting"
    st.session_state.user_data = {}
    st.session_state.last_input = None
    st.session_state.itinerary_text = ""
    st.session_state.finalized = False

user_input = st.chat_input("Say something...")
if user_input:
    st.session_state.last_input = user_input.strip()

if st.session_state.chat_stage == "collecting":
    if "destination" not in st.session_state.user_data:
        st.chat_message("assistant").markdown("Where would you like to travel?")
        if st.session_state.last_input:
            st.session_state.user_data["destination"] = st.session_state.last_input
            st.session_state.last_input = None
            st.rerun()

    elif "days" not in st.session_state.user_data:
        st.chat_message("assistant").markdown("How many days will your trip be?")
        if st.session_state.last_input:
            try:
                st.session_state.user_data["days"] = int(st.session_state.last_input)
                st.session_state.last_input = None
                st.rerun()
            except ValueError:
                st.chat_message("assistant").markdown("Please enter a valid number.")
                st.session_state.last_input = None

    elif "budget" not in st.session_state.user_data:
        st.chat_message("assistant").markdown("What's your budget? (low-range, mid-range, luxury)")
        if st.session_state.last_input:
            st.session_state.user_data["budget"] = st.session_state.last_input.lower().strip()
            st.session_state.last_input = None
            st.rerun()

    elif "departure" not in st.session_state.user_data:
        st.chat_message("assistant").markdown("Which city will you be departing from?")
        if st.session_state.last_input:
            st.session_state.user_data["departure"] = st.session_state.last_input
            st.session_state.chat_stage = "planning"
            st.session_state.last_input = None
            st.rerun()


def get_flight_info(departure, arrival):
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

    depart_date = (date.today() + timedelta(days=30)).isoformat()
    return_date = (date.today() + timedelta(days=37)).isoformat()

    params = {
        "engine": "google_flights",
        "departure_id": departure_code,
        "arrival_id": arrival_code,
        "outbound_date": depart_date,
        "return_date": return_date,
        "currency": "INR",
        "hl": "en",
        "gl": "in",
        "api_key": SERPAPI_API_KEY
    }

    try:
        results = GoogleSearch(params).get_dict()
        flights = results.get("best_flights", [])
        if not flights:
            flights = results.get("other_flights", [])

        if not flights or not isinstance(flights, list):
            return "❌ No flights found or invalid data structure from SerpAPI."

        lines = []
        for f in flights[:3]:
            price = f.get("price", "N/A")
            carrier = "/".join({seg.get("airline", "-") for seg in f.get("flights", [])})
            lines.append(f"- {carrier} | ₹{price:,}" if isinstance(price, int) else f"- {carrier} | ₹{price}")
        return "\n".join(lines)

    except Exception as e:
        return f"Exception occurred while fetching flights: {e}"


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
        "api_key": SERPAPI_API_KEY
    }
    props = GoogleSearch(params).get_dict().get("properties", [])
    if not props:
        return "No hotels found via SerpAPI."

    price_caps = {"low-range": 4000, "mid-range": 9000, "luxury": 90000}
    cap = price_caps.get(budget, 9000)
    shortlist = []

    for h in props:
        price = h.get("rate_per_night", {}).get("extracted_lowest")
        if price and price <= cap:
            shortlist.append(f"- {h['name']} | ₹{int(price):,}")
        if len(shortlist) == 3:
            break

    return "\n".join(shortlist) or "No hotels matched the chosen budget."


def generate_full_itinerary(prefs):
    flights_txt = get_flight_info(prefs['departure'], prefs['destination'])
    hotels_txt = get_hotel_info(prefs['destination'], prefs['budget'])

    flight_finder = Agent(role="Flight Finder",
                          goal="Fetch best flight options",
                          backstory="Flight search expert",
                          llm=llm)

    hotel_finder = Agent(role="Hotel Finder",
                         goal="Find best hotels",
                         backstory="Hotel deals expert",
                         llm=llm)

    itinerary_planner = Agent(role="Trip Planner",
                              goal="Craft day-wise travel plan",
                              backstory="Luxury itinerary expert",
                              llm=llm)

    flight_task = Task(description=f"""
        Here are the best flights from {prefs['departure']} to 
        {prefs['destination']}:
        {flights_txt}
    """, expected_output="Flight summary", agent=flight_finder)

    hotel_task = Task(description=f"""
        Here are 3 top hotels in {prefs['destination']} for a 
        {prefs['budget']} budget:
        {hotels_txt}
    """, expected_output="Hotel summary", agent=hotel_finder)

    itinerary_task = Task(description=f"""
    You are an expert travel planner. You must create a {prefs['days']}-day 
    itinerary for a luxury trip from {prefs['departure']} to 
    {prefs['destination']}.

    Here are the flight and hotel options fetched from real-time data:
    - FLIGHTS:
    {flights_txt}

    - HOTELS:
    {hotels_txt}

    Strict Instructions:
    - Do NOT invent any explanations or assumptions like "assuming 
      flight is booked", or hotel name clarifications.
    - Only use the flights and hotels as-is.
    - DO NOT explain or justify anything.
    - DO NOT add comments or introductions like "Here's your itinerary".
    - JUST write the itinerary in the format below.

    Format Strictly:
    Day 1:
    - Morning: ...
    - Afternoon: ...
    - Evening: ...
    Day 2:
    ...

    Only return the full itinerary in this exact format. Do not skip any day.
    """,
                          expected_output="Complete itinerary",
                          agent=itinerary_planner)

    crew = Crew(agents=[flight_finder, hotel_finder, itinerary_planner],
                tasks=[flight_task, hotel_task, itinerary_task])
    result = crew.kickoff()

    return flights_txt, hotels_txt, str(
        itinerary_task.output) if itinerary_task.output else "Itinerary generation failed."


if st.session_state.chat_stage == "planning" and not st.session_state.itinerary_text:
    with st.spinner("Crafting your perfect itinerary..."):
        flights, hotels, itinerary = generate_full_itinerary(st.session_state.user_data)
        st.session_state.flight_info = flights
        st.session_state.hotel_info = hotels
        st.session_state.itinerary_text = str(itinerary)
        st.rerun()

if st.session_state.last_input and str(st.session_state.itinerary_text).strip().startswith("Day"):
    editor = Agent(
        role="Itinerary Editor",
        goal="Apply user changes",
        backstory="Experienced editor",
        llm=llm
    )

    edit_task = Task(
        description=f"""
    You are an expert itinerary editor.

    The user wants to make a change to this day-wise travel itinerary:

    --- Original Itinerary ---
    {str(st.session_state.itinerary_text)}

    --- User Request ---
    {st.session_state.last_input}

    Your task:

    1. Identify the *exact day and time block* mentioned by the user.
    2. Replace only that section with a realistic, location-appropriate activity.
    3. Leave the rest of the itinerary exactly unchanged.
    4. Return *only* the full updated itinerary in markdown format like below:

    Day 1:
    - Morning: ...
    - Afternoon: ...
    - Evening: ...

    VERY IMPORTANT:
    - Do NOT include any comments, thoughts, prefaces, explanations, or "thoughts".
    - Return ONLY the updated itinerary.
    """,
        expected_output="Updated itinerary",
        return_value=True,
        agent=editor
    )

    with st.spinner("Applying changes..."):
        crew = Crew(agents=[editor], tasks=[edit_task])
        updated = str(crew.kickoff())
        if updated.strip().lower().startswith("day"):
            st.session_state.itinerary_text = updated
        else:
            st.chat_message("assistant").markdown("Failed to apply the change correctly. Try rewording your input.")

        st.session_state.last_input = None
        st.rerun()

if st.session_state.itinerary_text:
    st.chat_message("assistant").markdown("**Flights:**")
    st.chat_message("assistant").markdown(st.session_state.flight_info)

    st.chat_message("assistant").markdown("**Hotels:**")
    st.chat_message("assistant").markdown(st.session_state.hotel_info)

    st.chat_message("assistant").markdown("**Itinerary:**")
    st.chat_message("assistant").markdown(st.session_state.itinerary_text)

    st.chat_message("assistant").markdown("Need any changes? Type them below.")
