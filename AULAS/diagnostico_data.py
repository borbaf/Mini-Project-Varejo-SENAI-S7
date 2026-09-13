import pandas as pd          
import unittest

class TestEstatistica(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"CL_FHL": [0, 1, 1, 2, 2, 2, 3, 4]})

    def test_media(self):
        est = estatistica_descritiva(self.df)
        self.assertAlmostEqual(est["media"], 1.875)

    def test_moda(self):
        est = estatistica_descritiva(self.df)
        self.assertEqual(est["moda"], 2)

    def test_min_max(self):
        est = estatistica_descritiva(self.df)
        self.assertEqual(est["minimo"], 0)
        self.assertEqual(est["maximo"], 4)

if __name__ == "__main__":
    unittest.main()