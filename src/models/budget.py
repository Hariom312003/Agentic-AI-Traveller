"""Budget and rewards domain models."""
from __future__ import annotations

from pydantic import BaseModel, Field, computed_field


class BudgetBreakdown(BaseModel):
    currency: str = "INR"
    flights: float = 0.0
    hotels: float = 0.0
    food: float = 0.0
    activities: float = 0.0
    shopping: float = 0.0
    local_transport: float = 0.0
    emergency_buffer: float = 0.0
    taxes_and_fees: float = 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total(self) -> float:
        return round(
            self.flights
            + self.hotels
            + self.food
            + self.activities
            + self.shopping
            + self.local_transport
            + self.emergency_buffer
            + self.taxes_and_fees,
            2,
        )

    def per_day(self, duration_days: int) -> float:
        if duration_days <= 0:
            return 0.0
        return round(self.total / duration_days, 2)

    def as_chart_data(self) -> dict[str, float]:
        return {
            "Flights": self.flights,
            "Hotels": self.hotels,
            "Food": self.food,
            "Activities": self.activities,
            "Shopping": self.shopping,
            "Local Transport": self.local_transport,
            "Emergency Buffer": self.emergency_buffer,
            "Taxes & Fees": self.taxes_and_fees,
        }


class RewardRecommendation(BaseModel):
    category: str  # e.g. "Hotel Booking", "Dining", "Flights"
    instrument: str  # e.g. "Travel rewards credit card (example)"
    reason: str
    estimated_savings: float = 0.0
    currency: str = "INR"
    is_illustrative: bool = True
    """True unless backed by a live, current card-comparison data source.
    The dataset shipped with this project is illustrative reference data —
    see data/destinations/../rewards catalogue and docs/architecture.md for
    why we didn't fabricate specific real-world card offers."""


class RewardsSummary(BaseModel):
    recommendations: list[RewardRecommendation] = Field(default_factory=list)
    total_estimated_savings: float = 0.0
    currency: str = "INR"
    disclaimer: str = (
        "Card names and offers shown are illustrative examples for planning "
        "purposes only. Verify current terms with your card issuer before travel."
    )
