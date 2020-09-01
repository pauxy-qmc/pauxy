import numpy
import pytest
from pauxy.utils.linalg import column_pivoted_qr


@pytest.mark.unit
def test_column_pivoted_qr():
    A = numpy.random.randn(10,10)
    Q, D, T = column_pivoted_qr(A)
    Arecon = numpy.dot(numpy.dot(Q, numpy.diag(D)), T)
    assert numpy.linalg.norm(Arecon-A) == pytest.approx(0.0)
