"""Ingestion constants kept separate from normalization logic for maintainability."""

CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "econ": ("econ", "inflation", "fed", "rate", "gdp", "cpi", "jobs", "unemployment"),
    "policy": ("election", "senate", "house", "president", "law", "tariff"),
    "weather": ("weather", "temperature", "hurricane", "rain", "snow"),
    "sports": ("nba", "nfl", "mlb", "nhl", "soccer", "game"),
}
