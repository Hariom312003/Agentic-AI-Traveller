from __future__ import annotations

from pydantic import BaseModel, Field

class KnowledgeAttraction(BaseModel):
    name: str
    category: str = "attraction"
    description: str = ""
    recommended_duration: str | None = None
    budget_category: str | None = None
    best_time: str | None = None
    transport_tips: str | None = None
    tags: list[str] = Field(default_factory=list)
    latitude: float | None = None
    longitude: float | None = None
    address: str | None = None
    map_link: str | None = None

class KnowledgeRestaurant(BaseModel):
    name: str
    category: str = "food"
    description: str = ""
    recommended_duration: str | None = None
    budget_category: str | None = None
    tags: list[str] = Field(default_factory=list)
    latitude: float | None = None
    longitude: float | None = None
    address: str | None = None
    map_link: str | None = None

class KnowledgeHiddenGem(BaseModel):
    name: str
    category: str = "attraction"
    description: str = ""
    recommended_duration: str | None = None
    budget_category: str | None = None
    tags: list[str] = Field(default_factory=list)
    latitude: float | None = None
    longitude: float | None = None
    address: str | None = None
    map_link: str | None = None

class KnowledgeNightlife(BaseModel):
    name: str
    category: str = "nightlife"
    description: str = ""
    recommended_duration: str | None = None
    budget_category: str | None = None
    tags: list[str] = Field(default_factory=list)
    latitude: float | None = None
    longitude: float | None = None
    address: str | None = None
    map_link: str | None = None

class KnowledgeShopping(BaseModel):
    name: str
    category: str = "shopping"
    description: str = ""
    recommended_duration: str | None = None
    budget_category: str | None = None
    tags: list[str] = Field(default_factory=list)
    latitude: float | None = None
    longitude: float | None = None
    address: str | None = None
    map_link: str | None = None

class KnowledgeFestival(BaseModel):
    name: str
    category: str = "festival"
    description: str = ""
    best_time: str | None = None
    tags: list[str] = Field(default_factory=list)

class KnowledgeAdventureActivity(BaseModel):
    name: str
    category: str = "adventure"
    description: str = ""
    recommended_duration: str | None = None
    budget_category: str | None = None
    tags: list[str] = Field(default_factory=list)
    latitude: float | None = None
    longitude: float | None = None
    address: str | None = None
    map_link: str | None = None

class SafetyTips(BaseModel):
    general: str = ""
    water_safety: str | None = None
    transport_safety: str | None = None

class CultureEtiquette(BaseModel):
    overview: str = ""
    etiquette: str | None = None

class TransportInfo(BaseModel):
    getting_around: str = ""
    airport: str | None = None

class VisaInfo(BaseModel):
    domestic_indian_travellers: str | None = None
    foreign_nationals: str | None = None

class EmergencyContacts(BaseModel):
    police: str = "100"
    ambulance: str = "108"
    tourist_police_helpline: str | None = None

class WeatherInfo(BaseModel):
    best_time_to_visit: str = ""
    avoid: str | None = None

class DestinationKnowledge(BaseModel):
    destination: str
    attractions: list[KnowledgeAttraction] = Field(default_factory=list)
    restaurants: list[KnowledgeRestaurant] = Field(default_factory=list)
    hidden_gems: list[KnowledgeHiddenGem] = Field(default_factory=list)
    nightlife: list[KnowledgeNightlife] = Field(default_factory=list)
    shopping: list[KnowledgeShopping] = Field(default_factory=list)
    festivals: list[KnowledgeFestival] = Field(default_factory=list)
    adventure_activities: list[KnowledgeAdventureActivity] = Field(default_factory=list)
    local_tips: list[str] = Field(default_factory=list)
    safety: SafetyTips = Field(default_factory=SafetyTips)
    culture: CultureEtiquette = Field(default_factory=CultureEtiquette)
    transport: TransportInfo = Field(default_factory=TransportInfo)
    visa_information: VisaInfo = Field(default_factory=VisaInfo)
    emergency_contacts: EmergencyContacts = Field(default_factory=EmergencyContacts)
    weather: WeatherInfo = Field(default_factory=WeatherInfo)
