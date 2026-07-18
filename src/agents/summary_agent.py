"""
Summary Agent.

Terminal node for both the "plan" and "refine" graphs. Generates a comprehensive
AI Trip Summary from the validated itinerary and assembles the final JSON shape.
"""
from __future__ import annotations

import logging
from typing import Any

from src.agents.common import generate_structured
from src.models.state import TripState
from src.models.trip_summary import (
    TripSummary, TripOverview, TripHighlights, BudgetSummary, QuickStatistics,
    WeatherOverview, FoodRecommendations, TransportationSummary, ImportantTravelTips
)
from src.monitoring.logging_config import get_logger
from src.monitoring.telemetry import track_agent

logger = get_logger(__name__)


def build_fallback_summary(itinerary: Any, budget: Any, style: str, interests: str) -> TripSummary:
    """Build a rich, destination-specific fallback TripSummary object when LLM calls fail."""
    destination = (itinerary.destination if itinerary else None) or "your destination"
    duration_days = itinerary.duration_days if itinerary else 4
    
    # 1. Gather statistics from itinerary
    num_attractions = 0
    num_museums = 0
    num_parks = 0
    num_restaurants = 0
    num_shopping = 0
    num_adventure = 0
    num_cultural = 0
    all_activities = []
    
    if itinerary:
        for day in itinerary.days:
            for slot in ("morning", "afternoon", "evening", "night"):
                for act in day.slot(slot):
                    all_activities.append(act)
                    cat = act.category.value if hasattr(act.category, "value") else str(act.category)
                    if cat in ("attraction", "culture", "nature", "adventure"):
                        num_attractions += 1
                    if "museum" in act.title.lower() or "museum" in act.description.lower():
                        num_museums += 1
                    if "park" in act.title.lower() or "park" in act.description.lower() or cat == "nature":
                        num_parks += 1
                    if cat == "food":
                        num_restaurants += 1
                    if cat == "shopping":
                        num_shopping += 1
                    if cat == "adventure":
                        num_adventure += 1
                    if cat == "culture":
                        num_cultural += 1

    # Heuristics based on destination name
    dest_lower = destination.lower()
    if "goa" in dest_lower:
        best_time = "November to February"
        weather_desc = "Warm and Tropical"
        temp = "28°C"
        rain_prob = "Low"
        local_foods = ["Fish Curry", "Bebinca", "Feni", "Prawn Balchao", "Chicken Xacuti"]
        restaurants = ["Fisherman's Wharf", "Curlies", "Britto's", "Gunpowder"]
        veg_options = "Good options in most places, active beach shacks offer customized menus."
        sunset = "Anjuna Beach"
        photography = ["Fort Aguada", "Chapora Fort", "Fontainhas (Latin Quarter)", "Dudhsagar Falls"]
        tips = {
            "customs": "Dress modestly when entering religious sites and temples. Avoid bikinis outside beaches.",
            "safety": "Avoid swimming at beaches during high tide, monsoon, or post-sunset.",
            "currency": "Indian Rupee (INR)",
            "scams": "Be cautious of taxi drivers overcharging; pre-book cabs or use GoaMiles where possible.",
            "tipping": "Tipping 10% is standard in dine-in restaurants.",
            "internet": "Good 4G coverage; most cafes and beach shacks offer free Wi-Fi.",
            "sim": "Airtel or Jio offer best local coverage.",
            "laws": "Strict laws against drug usage and driving under the influence.",
            "etiquette": "Greet locals with a smile; ask before taking photos of individuals."
        }
    elif "tokyo" in dest_lower:
        best_time = "March to May (Sakura season) & September to November"
        weather_desc = "Mild and Pleasant"
        temp = "16°C"
        rain_prob = "Low to Moderate"
        local_foods = ["Sushi", "Ramen", "Tempura", "Yakitori", "Tonkatsu"]
        restaurants = ["Ichiran Ramen", "Sukiyabashi Jiro", "Tsunahachi Tempura"]
        veg_options = "Requires planning; download HappyCow. Specialized vegetarian/vegan cafes are growing."
        sunset = "Shibuya Sky"
        photography = ["Shibuya Crossing", "Senso-ji Temple", "Tokyo Tower", "Meiji Shrine"]
        tips = {
            "customs": "Bow when greeting. Do not tip in Japan. Avoid talking loudly on trains.",
            "safety": "Very safe, even at night. Watch for bicycle traffic on sidewalks.",
            "currency": "Japanese Yen (JPY)",
            "scams": "Avoid touts in Kabukicho promising cheap drinks or hostess clubs.",
            "tipping": "No tipping; it is considered disrespectful. Exceptional service is covered by the bill.",
            "internet": "Excellent Wi-Fi in train stations, convenience stores, and hotels.",
            "sim": "eSIM or pocket Wi-Fi is highly recommended and should be ordered in advance.",
            "laws": "Carry passport at all times (it is the law for foreigners). Strict drug laws.",
            "etiquette": "Do not eat while walking. Always walk on the designated side of escalators."
        }
    elif "paris" in dest_lower:
        best_time = "April to June & September to October"
        weather_desc = "Mild"
        temp = "14°C"
        rain_prob = "Moderate"
        local_foods = ["Croissant", "Escargot", "Crepes", "Macarons", "Coq au Vin"]
        restaurants = ["Le Comptoir du Relais", "Angelina", "Bouillon Chartier", "Le Relais de l'Entrecôte"]
        veg_options = "Very good availability in modern bistros and bakeries; traditional spots are limited."
        sunset = "Seine River Cruise / Eiffel Tower summit"
        photography = ["Eiffel Tower (Trocadéro)", "Louvre Pyramid", "Montmartre & Sacré-Cœur", "Arc de Triomphe"]
        tips = {
            "customs": "Greet shopkeepers with 'Bonjour' or 'Bonsoir'. It is considered rude not to say it.",
            "safety": "Watch for pickpockets in crowded tourist spots, metro lines, and train stations.",
            "currency": "Euro (EUR)",
            "scams": "Beware of 'ring trick' or gold ring scammers, petition signers, and string bracelet makers.",
            "tipping": "Service charge is included, but rounding up 5-10% for good service is polite.",
            "internet": "Free Wi-Fi in public parks, libraries, and cafes.",
            "sim": "Orange, SFR, or Bouygues offer great temporary tourist SIMs.",
            "laws": "Do not wear full-face veils in public places. Carrying ID is required.",
            "etiquette": "Keep your voice low in public transport and traditional cafes."
        }
    elif "manali" in dest_lower:
        best_time = "October to June (avoid monsoon in July-August)"
        weather_desc = "Cool and Mountainous"
        temp = "12°C"
        rain_prob = "Low to Moderate"
        local_foods = ["Siddu", "Trout Fish", "Kadhi Chawal", "Mittha", "Babru"]
        restaurants = ["Johnson's Cafe", "Cafe 1947", "Chopsticks", "The Lazy Dog"]
        veg_options = "Excellent North Indian and Tibetan vegetarian food widely available."
        sunset = "Solang Valley viewpoint"
        photography = ["Hadimba Temple", "Jogini Waterfall", "Rohtang Pass", "Solang Valley"]
        tips = {
            "customs": "Respect local pahadi customs. Dress warmly and wear comfortable hiking shoes.",
            "safety": "Be careful on narrow mountain roads. Check for landslide warnings during rain.",
            "currency": "Indian Rupee (INR)",
            "scams": "Confirm taxi/horse riding prices in Solang Valley in advance. Buy saffron from official stores.",
            "tipping": "10% is appreciated in tourist restaurants but not mandatory.",
            "internet": "Airtel/Jio work well in main areas, but remote valleys may lose signal.",
            "sim": "Local Indian prepaid SIM works well. Foreign SIMs have poor reception.",
            "laws": "Do not litter; Manali has a strict ban on plastic bags and littering.",
            "etiquette": "Always ask before taking photos of local children, elders, or religious ceremonies."
        }
    elif "bali" in dest_lower:
        best_time = "April to October (Dry season)"
        weather_desc = "Warm and Sunny"
        temp = "27°C"
        rain_prob = "Low (dry season)"
        local_foods = ["Nasi Goreng", "Sate Lilit", "Babi Guling", "Gado-Gado", "Nasi Campur"]
        restaurants = ["Naughty Nuri's", "Locavore", "Potato Head Beach Club", "Clear Cafe Ubud"]
        veg_options = "Outstanding vegetarian/vegan scene, especially in Ubud and Canggu."
        sunset = "Uluwatu Temple cliffs / Tanah Lot"
        photography = ["Tegallalang Rice Terrace", "Pura Lempuyang (Gates of Heaven)", "Ubud Monkey Forest", "Nusa Penida"]
        tips = {
            "customs": "Dress respectfully in temples (sarong is required, usually available to rent).",
            "safety": "Drink bottled water only (avoid 'Bali belly'). Ride scooters carefully and wear helmets.",
            "currency": "Indonesian Rupiah (IDR)",
            "scams": "Use only official money changers (BMC/Authorized) to avoid count scams. Confirm taxi meter.",
            "tipping": "Tipping is not traditional, but 5-10% is widely appreciated in tourist spots.",
            "internet": "Good coverage in south Bali and Ubud; remote areas vary. Most cafes have free Wi-Fi.",
            "sim": "Telkomsel Tourist SIM is recommended and can be bought at the airport.",
            "laws": "Very severe penalties for drug offenses. Respect local ceremonial processions (do not block).",
            "etiquette": "Avoid using your left hand to hand over items. Do not step on temple offerings (canang sari)."
        }
    else:
        best_time = "Year-round"
        weather_desc = "Pleasant"
        temp = "22°C"
        rain_prob = "Low to Moderate"
        local_foods = ["Signature local specialty", "Popular regional dishes", "Street market snacks"]
        restaurants = ["Top-rated local cafe", "Famous street food stall", "Scenic dining terrace"]
        veg_options = "Vegetarian options are available, check menus in advance."
        sunset = "Local scenic viewpoint"
        photography = ["City center landmarks", "Scenic local parks", "Historic cathedral or temple"]
        tips = {
            "customs": "Respect local culture, dress codes, and religious traditions.",
            "safety": "Keep personal belongings secure; walk in well-lit areas at night.",
            "currency": "Local currency",
            "scams": "Confirm taxi prices in advance. Be cautious of unsolicited guides.",
            "tipping": "10% is standard if service is not included.",
            "internet": "Public Wi-Fi available in central tourist areas and cafes.",
            "sim": "Purchase local SIM at airport or central telecom stores.",
            "laws": "Observe all local traffic, recycling, and public safety rules.",
            "etiquette": "Be polite and friendly with local service workers. Respect photography signs."
        }
        
    # Build Overview
    overview = TripOverview(
        destination=destination,
        duration_days=duration_days,
        travel_style=(style or "standard").capitalize(),
        estimated_budget=f"{budget.total:,.0f} {budget.currency}" if budget else "Cost Varies",
        best_time_to_visit=best_time,
        total_attractions=max(num_attractions, duration_days * 2),
        cities_covered=[destination],
        total_walking_distance=f"~{duration_days * 3.5:.1f} km",
        estimated_daily_travel="1–2 hours",
        recommended_transport="Local Taxi + Walking",
        weather=weather_desc,
        difficulty="Easy to Moderate" if duration_days > 3 else "Easy",
        overall_trip_rating="⭐⭐⭐⭐⭐"
    )
    
    # Rationale dynamic text
    top_atts = [act.title for act in all_activities if (act.category.value if hasattr(act.category, "value") else str(act.category)) in ("attraction", "culture", "nature", "adventure")][:4]
    if not top_atts:
        top_atts = ["historical sites", "scenic spots", "local markets"]
    
    ai_summary = (
        f"This offline fallback travel summary for {destination} is generated based on your requested {style} travel style "
        f"and key interests in {interests or 'general exploration'}. The curated plan spans {duration_days} days, balancing active sightseeing "
        f"with leisure time. You will explore top attractions such as {', '.join(top_atts[:3])}. Daily schedules are optimized "
        f"geographically to minimize transit times. The food scene covers famous spots and must-try local specialties, keeping budget constraints "
        f"and style preferences in view. Prepare for a rich experience in {destination} with comfortable pacing throughout."
    )
    
    # Highlights
    highlights = TripHighlights(
        top_attractions=top_atts,
        hidden_gems=[act.title for act in all_activities if (act.category.value if hasattr(act.category, "value") else str(act.category)) == "nature"] or ["Scenic local hideaway", "Quiet neighborhood street"],
        best_restaurants=[act.title for act in all_activities if (act.category.value if hasattr(act.category, "value") else str(act.category)) == "food"] or restaurants,
        signature_local_foods=local_foods,
        best_sunset_spot=sunset,
        best_photography_locations=photography,
        adventure_activities=[act.title for act in all_activities if (act.category.value if hasattr(act.category, "value") else str(act.category)) == "adventure"] or ["Local nature walking trail"],
        shopping_streets=[act.title for act in all_activities if (act.category.value if hasattr(act.category, "value") else str(act.category)) == "shopping"] or ["Traditional local marketplace"],
        nightlife_areas=[act.title for act in all_activities if (act.category.value if hasattr(act.category, "value") else str(act.category)) == "nightlife"] or ["Central lounge district"],
        must_try_experiences=[f"Experience the authentic local culture of {destination}"]
    )
    
    # Budget Summary
    b_summary = BudgetSummary(
        flights=f"{budget.flights:,.0f} {budget.currency}" if (budget and budget.flights) else "Varies",
        hotels=f"{budget.hotels:,.0f} {budget.currency}" if (budget and budget.hotels) else "Varies",
        food=f"{budget.food:,.0f} {budget.currency}" if (budget and budget.food) else "Varies",
        transport=f"{budget.local_transport:,.0f} {budget.currency}" if (budget and budget.local_transport) else "Varies",
        activities=f"{budget.activities:,.0f} {budget.currency}" if (budget and budget.activities) else "Varies",
        shopping=f"{budget.shopping:,.0f} {budget.currency}" if (budget and budget.shopping) else "Varies",
        emergency_buffer=f"{budget.emergency_buffer:,.0f} {budget.currency}" if (budget and budget.emergency_buffer) else "Varies",
        taxes=f"{budget.taxes_and_fees:,.0f} {budget.currency}" if (budget and budget.taxes_and_fees) else "Varies",
        total_cost=f"{budget.total:,.0f} {budget.currency}" if budget else "Varies",
        budget_utilization="95%",
        savings_suggestions=[
            "Book hotels and flights 3-4 weeks in advance for better rates.",
            "Use public transit or walking instead of private cabs where feasible.",
            "Eat at local diners or street markets for lunch to save on dining costs."
        ]
    )
    
    # Quick Statistics
    statistics = QuickStatistics(
        number_of_attractions=max(num_attractions, duration_days * 2),
        number_of_museums=num_museums,
        number_of_parks=num_parks,
        number_of_restaurants=max(num_restaurants, duration_days),
        number_of_shopping_areas=num_shopping,
        number_of_adventure_activities=num_adventure,
        number_of_cultural_experiences=num_cultural,
        total_estimated_travel_time="1.5 hours daily",
        total_walking_distance=f"~{duration_days * 3.5:.1f} km",
        average_daily_cost=f"{(budget.total / duration_days):,.0f} {budget.currency}" if budget else "N/A",
        average_daily_activity_duration="5-6 hours"
    )
    
    # Weather
    weather = WeatherOverview(
        average_temperature=temp,
        rain_probability=rain_prob,
        packing_suggestions=["Comfortable walking shoes", "Sunscreen & Sunglasses", "Appropriate clothing for the weather"],
        clothing_recommendations="Lightweight, breathable clothes; carrying a light jacket is suggested.",
        weather_warnings="None"
    )
    
    # Food Recs
    food = FoodRecommendations(
        must_try_local_foods=local_foods,
        famous_restaurants=restaurants,
        vegetarian_options=veg_options,
        street_food_recommendations=[local_foods[0] if local_foods else "Traditional snacks"],
        local_specialties=local_foods
    )
    
    # Transportation
    transport = TransportationSummary(
        airport_transfer="Available via prepaid taxi or bus service",
        local_transport="App-based cabs, local buses, auto-rickshaws",
        metro="N/A (varies by city)",
        taxi="Widely available, confirm meter usage or agree on price upfront",
        ride_sharing="Available in major areas",
        walking="Excellent for central markets and historic districts",
        estimated_daily_travel_time="1–2 hours"
    )
    
    # Tips
    tips_model = ImportantTravelTips(
        local_customs=tips["customs"],
        safety_advice=tips["safety"],
        currency=tips["currency"],
        emergency_contacts="112 (Universal Emergency Number)",
        common_scams=tips["scams"],
        tipping_etiquette=tips["tipping"],
        internet_availability=tips["internet"],
        sim_card_suggestions=tips["sim"],
        local_laws=tips["laws"],
        cultural_etiquette=tips["etiquette"]
    )
    
    # Reasoning
    reasoning = (
        f"This itinerary is constructed to organize the main sights of {destination} "
        f"into geographic clusters. Days are balanced with morning sightseeing when temperatures are cooler, "
        f"afternoon dining/resting, and evening leisure. Fallback routing schedules real places from the seed knowledge base."
    )
    
    return TripSummary(
        overview=overview,
        ai_summary=ai_summary,
        highlights=highlights,
        budget=b_summary,
        statistics=statistics,
        weather=weather,
        food=food,
        transport=transport,
        tips=tips_model,
        ai_reasoning=reasoning
    )


def build_final_response(state: TripState) -> dict:
    """Pure assembly function — used both by the Summary Agent node (during
    a normal graph run) and directly by the API's rollback endpoint (which
    needs to render a response for a historical checkpoint that may
    predate the Summary Agent ever running in that branch)."""
    itinerary = state.get("itinerary")
    budget = state.get("budget_breakdown")
    rewards = state.get("rewards_summary")
    validation = state.get("validation_report")
    trip_summary = state.get("trip_summary")
    trace = state.get("execution_trace", [])

    return {
        "itinerary": itinerary.model_dump(mode="json") if itinerary else None,
        "budget": budget.model_dump(mode="json") if budget else None,
        "rewards": rewards.model_dump(mode="json") if rewards else None,
        "validation": validation.model_dump(mode="json") if validation else None,
        "trip_summary": trip_summary.model_dump(mode="json") if trip_summary else None,
        "explainability": {
            "execution_trace": [r.model_dump(mode="json") for r in trace],
            "total_agents_run": len(trace),
            "total_latency_ms": round(sum(r.latency_ms or 0 for r in trace), 1),
            "providers_used": sorted({r.llm_provider for r in trace if r.llm_provider}),
        },
        "warnings": state.get("errors", []),
    }


def run_summary_agent(state: TripState) -> dict:
    with track_agent("summary_agent") as handle:
        itinerary = state.get("itinerary")
        destination = (itinerary.destination if itinerary else None) or "your destination"
        style = (state.get("trip_request").travel_style if state.get("trip_request") else None) or "standard"
        interests = ", ".join((state.get("trip_request").interests if state.get("trip_request") else None) or []) or "none"

        itinerary_text = ""
        if itinerary:
            lines = []
            for day in itinerary.days:
                lines.append(f"Day {day.day_number}: {day.theme}")
                for slot in ("morning", "afternoon", "evening", "night"):
                    for act in day.slot(slot):
                        lines.append(
                            f"  - [{slot.capitalize()}] {act.title}: {act.description}\n"
                            f"    Category: {act.category.value if hasattr(act.category, 'value') else str(act.category)}, Cost: {act.estimated_cost} {act.currency}, "
                            f"    Duration: {act.duration_minutes} mins, Coordinates: ({act.latitude}, {act.longitude})"
                        )
            itinerary_text = "\n".join(lines)

        budget = state.get("budget_breakdown")
        budget_text = ""
        if budget:
            budget_text = (
                f"Total Estimated Cost: {budget.total} {budget.currency}\n"
                f"Cost items: {budget.as_chart_data()}"
            )

        system_prompt = (
            "You are the Summary agent of a travel planning system. You are a senior travel consultant.\n"
            "Your job is to read the final validated itinerary and generate a comprehensive, highly professional, and personalized Trip Summary matching the requested JSON schema.\n"
            "Ensure your natural-language AI summary is a detailed, beautiful paragraph (200-400 words) describing the destination, selected attractions rationale, transit flow, pace, and expected experiences.\n"
            "Make all details (weather, packing tips, custom warnings, local foods, transit options) highly specific to the actual destination and itinerary.\n"
            "Populate all fields in the TripSummary schema accurately."
        )

        user_prompt = (
            f"Please review this final validated travel itinerary for '{destination}':\n\n"
            f"{itinerary_text}\n\n"
            f"Budget Breakdown:\n{budget_text}\n\n"
            f"Travel Request Details:\n"
            f"- Travel Style: {style}\n"
            f"- Interests: {interests}\n\n"
            f"Generate a professional, fully complete TripSummary object."
        )

        trip_summary = None
        if itinerary:
            try:
                trip_summary, _ = generate_structured(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    schema=TripSummary,
                    temperature=0.3,
                )
            except Exception as exc:
                logger.error(f"Failed to generate AI trip summary dynamically: {exc}")
                logger.info("Generating fallback trip summary")
                trip_summary = build_fallback_summary(itinerary, budget, style, interests)

        # Update state slice
        new_state_slice = {
            "trip_summary": trip_summary,
        }

        # Build final response combining updated state
        updated_state = dict(state)
        updated_state.update(new_state_slice)
        final_response = build_final_response(updated_state)

        handle.record.reasoning_summary = "AI Trip Summary generated and final response compiled"

        return {
            "trip_summary": trip_summary,
            "final_response": final_response,
            "execution_trace": [handle.record]
        }
