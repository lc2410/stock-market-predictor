"""Shared pytest fixtures for backend tests."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def dummy_stock_data():
    """Generates 1,300 days of fake stock data for model testing."""
    dates = pd.date_range(start="2018-01-01", periods=1300, freq="B")
    np.random.seed(42)
    
    df = pd.DataFrame({
        "Close": np.linspace(100, 200, 1300) + np.random.normal(0, 2, 1300),
        "Volume": np.random.randint(100000, 500000, 1300),
        "High": np.linspace(105, 205, 1300),
        "Low": np.linspace(95, 195, 1300),
        "Dividends": 0.0
    }, index=dates)
    
    div_indices = np.random.choice(1300, 30, replace=False)
    df.loc[df.index[div_indices], "Dividends"] = np.random.uniform(0.1, 0.5, 30)
        
    return df
