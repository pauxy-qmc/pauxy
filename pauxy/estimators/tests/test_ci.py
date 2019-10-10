import pytest
from pauxy.systems.ueg import UEG
from pauxy.estimators.ci import simple_fci

def test_ueg():
    sys = UEG({'rs': 2, 'nup': 2, 'ndown': 2, 'ecut': 0.5})
    sys.ecore = 0
    eig, evec = simple_fci(sys)
    assert len(eig) == 441
    assert eig[0] == pytest.approx(1.327088181107)
    assert eig[231] == pytest.approx(2.883365264420)
    assert eig[424] == pytest.approx(3.039496944900)
    assert eig[-1] == pytest.approx(3.207573492596)
