import unittest
import numpy
from pauxy.systems.ueg import UEG
from pauxy.estimators.ci import simple_fci

class TestCI(unittest.TestCase):

    def test_ueg(self):
        sys = UEG({'rs': 2, 'nup': 2, 'ndown': 2, 'ecut': 0.5})
        sys.ecore = 0
        eig, evec = simple_fci(sys)
        self.assertEqual(len(eig),441)
        self.assertAlmostEqual(eig[0], 1.327088181107)
        self.assertAlmostEqual(eig[231], 2.883365264420)
        self.assertAlmostEqual(eig[424], 3.039496944900)
        self.assertAlmostEqual(eig[-1],  3.207573492596)

if __name__ == '__main__':
    unittest.main()
