import unittest
import nn
import nn2
import data_reader

class TestNN(unittest.TestCase):
    def test_nn_exists(self):
        self.assertTrue(hasattr(nn, "__file__"))

class TestNN2(unittest.TestCase):
    def test_nn2_exists(self):
        self.assertTrue(hasattr(nn2, "__file__"))

class TestDataReader(unittest.TestCase):
    def test_data_reader_exists(self):
        self.assertTrue(hasattr(data_reader, "__file__"))

if __name__ == "__main__":
    unittest.main()
