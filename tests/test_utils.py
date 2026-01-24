"""
Unit tests for utility functions
"""

import pytest
import numpy as np
import pandas as pd
from src.utils import normalize_data


def test_normalize_data():
    """Test data normalization."""
    data = pd.Series([1, 2, 3, 4, 5])
    normalized = normalize_data(data)
    
    assert normalized.min() == 0
    assert normalized.max() == 1
    assert len(normalized) == len(data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
