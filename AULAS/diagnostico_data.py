import unittest

class TestAgrupamento(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "CL_GENERO": ["M", "M", "F", "M"],
            "PR_CAT": ["BEBIDAS", "ALIMENTOS", "BEBIDAS", "BEBIDAS"],
            "DATA": pd.to_datetime(["01/02/2019", "01/02/2019",
                                    "01/03/2019", "15/03/2019"],
                                   dayfirst=True),
        })

    def test_genero(self):
        s = agrupar_por_genero(self.df)
        self.assertEqual(s["M"], 3)
        self.assertEqual(s["F"], 1)

    def test_categoria(self):
        s = agrupar_por_categoria(self.df)
        self.assertEqual(s["BEBIDAS"], 3)
        self.assertEqual(s["ALIMENTOS"], 1)

    def test_mes(self):
        s = agrupar_por_mes(self.df)
        self.assertEqual(s["2019-02"], 2)
        self.assertEqual(s["2019-03"], 2)

if __name__ == "__main__":
    unittest.main()