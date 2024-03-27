import numpy as np
import pytest
from TV_utils import fl_dp_weight, fl_dp


def test_fl_dp_weight():
    """Test the fl_dp_weight function."""
    y = np.array([1, 2, 3, 4, 5])
    a1 = 1
    lamb1 = 2

    a2 = 2
    lamb2 = 2
    y_modified = fl_dp_weight(y, a1, lamb1)

    assert (y_modified == [0, 1, 1, 2, 2]).all()

    y_modified = fl_dp_weight(y, a2, lamb2)

    assert (y_modified == [0, 2, 3, 4, 5]).all()


def test_fl_dp():
    """Test the fl_dp function."""
    y = np.array([1, 2, 3, 4, 5])
    lamb1 = 1
    lamb2 = 4
    y_modified = fl_dp(y, lamb1)

    assert (y_modified == [0, 1, 1, 1, 1]).all()

    y_modified = fl_dp(y, lamb2)

    assert (y_modified == [-3, -2,-1, 0, 1]).all()