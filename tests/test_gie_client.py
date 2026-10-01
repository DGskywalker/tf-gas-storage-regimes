"""Unit tests for GIEClient and AGSI+ ingestion."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.data.gie_client import GIEClient
from src.data.schemas import validate_storage_dataframe


def test_gie_client_initialization(tmp_dir: Path) -> None:
    """Test client instantiation with custom cache dir."""
    client = GIEClient(api_key="test_key", cache_dir=tmp_dir)
    assert client.api_key == "test_key"
    assert client.cache_dir == tmp_dir
    assert client.session.headers.get("x-key") == "test_key"


def test_fetch_eu_storage_calibrated(tmp_dir: Path) -> None:
    """Test fallback calibrated EU storage generation and validation."""
    client = GIEClient(api_key=None, cache_dir=tmp_dir, use_mock_fallback=True)
    df = client.fetch_eu_storage(start="2022-01-01", end="2022-03-31")

    assert not df.empty
    assert len(df) == 90
    assert "gas_day" in df.columns
    assert "full" in df.columns
    assert "gas_in_storage" in df.columns
    assert (df["country_code"] == "EU").all()

    # Boundary assertions
    assert (df["full"] >= 0.0).all() and (df["full"] <= 105.0).all()
    assert (df["gas_in_storage"] >= 0.0).all()

    # Pandera validation should pass without error
    validated = validate_storage_dataframe(df)
    assert len(validated) == len(df)


def test_fetch_country_storage(tmp_dir: Path) -> None:
    """Test country-level storage fetching for Germany."""
    client = GIEClient(api_key=None, cache_dir=tmp_dir, use_mock_fallback=True)
    df = client.fetch_country_storage(country_code="DE", start="2023-01-01", end="2023-02-01")

    assert not df.empty
    assert (df["country_code"] == "DE").all()
    assert df["working_gas_volume"].iloc[0] == 250.0


def test_fetch_facility_metadata(tmp_dir: Path) -> None:
    """Test facility metadata retrieval."""
    client = GIEClient(api_key=None, cache_dir=tmp_dir)
    df_meta = client.fetch_facility_metadata()

    assert not df_meta.empty
    assert "facility_name" in df_meta.columns
    assert "operator" in df_meta.columns
    assert "working_gas_twh" in df_meta.columns
    assert "Rehden" in df_meta["facility_name"].values


@patch("requests.Session.get")
def test_paginated_api_handling(mock_get: MagicMock, tmp_dir: Path) -> None:
    """Test that pagination correctly loops through pages."""
    mock_resp_1 = MagicMock()
    mock_resp_1.status_code = 200
    mock_resp_1.json.return_value = {
        "data": [{"gasDayStart": "2023-01-01", "gasInStorage": 100.0, "full": 50.0}],
        "last_page": 2,
    }

    mock_resp_2 = MagicMock()
    mock_resp_2.status_code = 200
    mock_resp_2.json.return_value = {
        "data": [{"gasDayStart": "2023-01-02", "gasInStorage": 102.0, "full": 51.0}],
        "last_page": 2,
    }

    mock_get.side_effect = [mock_resp_1, mock_resp_2]

    client = GIEClient(api_key="valid_key", cache_dir=tmp_dir, use_mock_fallback=False)
    df = client._fetch_paginated_api("EU", "2023-01-01", "2023-01-02")

    assert len(df) == 2
    assert mock_get.call_count == 2
