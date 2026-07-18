import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.config import get_settings
from src.rag.acquisition import is_destination_cached, acquire_destination_knowledge
from src.models.destination_knowledge import DestinationKnowledge, KnowledgeAttraction

def test_is_destination_cached():
    # "Tokyo" is in the seed database, so it should be cached
    assert is_destination_cached("Tokyo") is True
    # A completely random name should not be cached
    assert is_destination_cached("Atlantis-City-XYZ") is False

@patch("src.rag.acquisition._fetch_page_text")
@patch("src.rag.acquisition.generate_structured")
@patch("src.rag.acquisition.VectorStore")
def test_acquire_destination_knowledge(mock_vector_store, mock_generate_structured, mock_fetch_text, tmp_path):
    settings = get_settings()
    # Temporarily override destinations_data_path to tmp_path to prevent polluting real cache
    original_path = settings.destinations_data_path
    settings.destinations_data_path = str(tmp_path)

    try:
        mock_fetch_text.return_value = "Test page content about Rome."
        
        # Mock structured model output
        mock_model = DestinationKnowledge(
            destination="Rome",
            attractions=[
                KnowledgeAttraction(
                    name="Colosseum",
                    category="attraction",
                    description="Historic Roman amphitheatre.",
                    recommended_duration="2 hours",
                    budget_category="Low-cost",
                    latitude=41.8902,
                    longitude=12.4922,
                    address="Piazza del Colosseo, 1, 00184 Roma RM, Italy",
                    map_link="https://www.openstreetmap.org/way/23048934"
                )
            ]
        )
        mock_generate_structured.return_value = (mock_model, MagicMock())

        # Mock vector store instance
        mock_store_instance = MagicMock()
        mock_vector_store.return_value = mock_store_instance

        # Call acquire
        chunks = acquire_destination_knowledge("Rome")

        # Verify
        assert len(chunks) > 0
        assert chunks[0].metadata["name"] == "Colosseum"
        assert chunks[0].metadata["latitude"] == 41.8902
        assert chunks[0].metadata["longitude"] == 12.4922

        # Verify cache file was written
        cache_file = tmp_path / "rome.json"
        assert cache_file.exists()

    finally:
        settings.destinations_data_path = original_path
