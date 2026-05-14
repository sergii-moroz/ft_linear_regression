import unittest
from pathlib import Path
from train import get_data_path


class TestGetDataPath(unittest.TestCase):

	def test_default_path(self):
		"""Test that default path is constructed correctly"""
		result = get_data_path()
		self.assertIsInstance(result, Path)
		self.assertEqual(result.name, 'ft_linear_regression_data.csv')
		self.assertEqual(result.parent.name, 'data')

	def test_custom_string_path(self):
		"""Test that custom string path is converted to Path"""
		custom_path = 'custom/path/data.csv'
		result = get_data_path(custom_path)
		self.assertIsInstance(result, Path)
		self.assertEqual(str(result), str(Path(custom_path)))

	def test_custom_path_object(self):
		"""Test that custom Path object is handled correctly"""
		custom_path = Path('another/path/file.csv')
		result = get_data_path(custom_path)
		self.assertIsInstance(result, Path)
		self.assertEqual(result, custom_path)

	def test_absolute_path(self):
		"""Test that absolute paths are preserved"""
		if Path.cwd().drive:
			absolute_path = Path.cwd() / 'test_data.csv'
		else:
			absolute_path = Path('/tmp/test_data.csv')
		result = get_data_path(str(absolute_path))
		self.assertIsInstance(result, Path)
		self.assertEqual(result, absolute_path)


if __name__ == '__main__':
	unittest.main()
