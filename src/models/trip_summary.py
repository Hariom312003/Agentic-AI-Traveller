from __future__ import annotations

from pydantic import BaseModel, Field


class TripOverview(BaseModel):
    destination: str
    duration_days: int
    travel_style: str
    estimated_budget: str
    best_time_to_visit: str
    total_attractions: int
    cities_covered: list[str] = Field(default_factory=list)
    total_walking_distance: str
    estimated_daily_travel: str
    recommended_transport: str
    weather: str
    difficulty: str
    overall_trip_rating: str


class TripHighlights(BaseModel):
    top_attractions: list[str] = Field(default_factory=list)
    hidden_gems: list[str] = Field(default_factory=list)
    best_restaurants: list[str] = Field(default_factory=list)
    signature_local_foods: list[str] = Field(default_factory=list)
    best_sunset_spot: str | None = None
    best_photography_locations: list[str] = Field(default_factory=list)
    adventure_activities: list[str] = Field(default_factory=list)
    shopping_streets: list[str] = Field(default_factory=list)
    nightlife_areas: list[str] = Field(default_factory=list)
    must_try_experiences: list[str] = Field(default_factory=list)


class BudgetSummary(BaseModel):
    flights: str
    hotels: str
    food: str
    transport: str
    activities: str
    shopping: str
    emergency_buffer: str
    taxes: str
    total_cost: str
    budget_utilization: str
    savings_suggestions: list[str] = Field(default_factory=list)


class QuickStatistics(BaseModel):
    number_of_attractions: int
    number_of_museums: int
    number_of_parks: int
    number_of_restaurants: int
    number_of_shopping_areas: int
    number_of_adventure_activities: int
    number_of_cultural_experiences: int
    total_estimated_travel_time: str
    total_walking_distance: str
    average_daily_cost: str
    average_daily_activity_duration: str


class WeatherOverview(BaseModel):
    average_temperature: str
    rain_probability: str
    packing_suggestions: list[str] = Field(default_factory=list)
    clothing_recommendations: str
    weather_warnings: str | None = None


class FoodRecommendations(BaseModel):
    must_try_local_foods: list[str] = Field(default_factory=list)
    famous_restaurants: list[str] = Field(default_factory=list)
    vegetarian_options: str
    street_food_recommendations: list[str] = Field(default_factory=list)
    local_specialties: list[str] = Field(default_factory=list)


class TransportationSummary(BaseModel):
    airport_transfer: str
    local_transport: str
    metro: str
    taxi: str
    ride_sharing: str
    walking: str
    estimated_daily_travel_time: str


class ImportantTravelTips(BaseModel):
    local_customs: str
    safety_advice: str
    currency: str
    emergency_contacts: str
    common_scams: str
    tipping_etiquette: str
    internet_availability: str
    sim_card_suggestions: str
    local_laws: str
    cultural_etiquette: str


class TripSummary(BaseModel):
    overview: TripOverview
    ai_summary: str = Field(description="200-400 words paragraph summarizing the trip.")
    highlights: TripHighlights
    budget: BudgetSummary
    statistics: QuickStatistics
    weather: WeatherOverview
    food: FoodRecommendations
    transport: TransportationSummary
    tips: ImportantTravelTips
    ai_reasoning: str
