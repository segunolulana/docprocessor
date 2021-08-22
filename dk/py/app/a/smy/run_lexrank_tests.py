import unittest
from run_lsa_lexrank import put_placeholders


class LsaLexrankTest(unittest.TestCase):
    def test_put_placeholders(self):
        text = """ history
* Test"""
        result = """ history.
*9 * Test"""
        self.assertEqual(put_placeholders(text), (1, result))


if __name__ == "__main__":
    unittest.main()
