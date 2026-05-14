import unittest
import pandas as pd
import numpy as np
from data_cleaner import (
	_check_required_columns,
	_convert_columns_to_numeric,
	_remove_invalid_rows,
	clean_data,
	MissingColumnsError,
	NoValidDataError
)

# ==========================================
# _check_required_columns
# ==========================================

class TestCheckRequiredColumns(unittest.TestCase):

	def test_all_columns_exist(self):
		"""Test when all required columns are present"""
		data = pd.DataFrame({'km': [1, 2, 3], 'price': [100, 200, 300]})
		# Should not raise any exception
		_check_required_columns(data, ['km', 'price'])

	def test_single_missing_column(self):
		"""Test when one required column is missing"""
		data = pd.DataFrame({'km': [1, 2, 3]})

		with self.assertRaises(MissingColumnsError) as context:
			_check_required_columns(data, ['km', 'price'])

		self.assertIn("Missing required column(s): ['price']", str(context.exception))

	def test_multiple_missing_columns(self):
		"""Test when multiple required columns are missing"""
		data = pd.DataFrame({'id': [1, 2, 3]})

		with self.assertRaises(MissingColumnsError) as context:
			_check_required_columns(data, ['km', 'price', 'year'])

		error_message = str(context.exception)
		self.assertIn("Missing required column(s):", error_message)
		self.assertIn("'km'", error_message)
		self.assertIn("'price'", error_message)
		self.assertIn("'year'", error_message)

	def test_empty_required_columns(self):
		"""Test when no columns are required"""
		data = pd.DataFrame({'km': [1, 2, 3], 'price': [100, 200, 300]})
		# Should not raise any exception
		_check_required_columns(data, [])

	def test_empty_dataframe(self):
		"""Test with empty DataFrame but required columns exist"""
		data = pd.DataFrame(columns=['km', 'price'])
		# Should not raise any exception
		_check_required_columns(data, ['km', 'price'])

	def test_case_sensitive_column_names(self):
		"""Test that column name matching is case-sensitive"""
		data = pd.DataFrame({'KM': [1, 2, 3], 'Price': [100, 200, 300]})

		with self.assertRaises(MissingColumnsError) as context:
			_check_required_columns(data, ['km', 'price'])

		self.assertIn("Missing required column(s):", str(context.exception))

# ==========================================
# _convert_columns_to_numeric
# ==========================================

class TestConvertColumnsToNumeric(unittest.TestCase):

	def test_all_numeric_values(self):
		"""Test conversion when all values are already numeric"""
		data = pd.DataFrame({'km': [10, 20, 30], 'price': [100, 200, 300]})
		result = _convert_columns_to_numeric(data.copy(), ['km', 'price'])

		pd.testing.assert_frame_equal(result, data)
		self.assertTrue(pd.api.types.is_numeric_dtype(result['km']))
		self.assertTrue(pd.api.types.is_numeric_dtype(result['price']))

	def test_string_numbers_conversion(self):
		"""Test conversion of string numbers to numeric"""
		data = pd.DataFrame({'km': ['10', '20', '30'], 'price': ['100', '200', '300']})
		result = _convert_columns_to_numeric(data.copy(), ['km', 'price'])

		self.assertTrue(pd.api.types.is_numeric_dtype(result['km']))
		self.assertTrue(pd.api.types.is_numeric_dtype(result['price']))
		self.assertEqual(result['km'].tolist(), [10.0, 20.0, 30.0])
		self.assertEqual(result['price'].tolist(), [100.0, 200.0, 300.0])

	def test_invalid_values_converted_to_nan(self):
		"""Test that non-numeric values are converted to NaN"""
		data = pd.DataFrame({'km': [10, 'invalid', 30], 'price': [100, 200, 'N/A']})

		result = _convert_columns_to_numeric(data.copy(), ['km', 'price'])

		self.assertTrue(pd.isna(result['km'].iloc[1]))
		self.assertTrue(pd.isna(result['price'].iloc[2]))
		self.assertEqual(result['km'].iloc[0], 10.0)
		self.assertEqual(result['price'].iloc[0], 100.0)

	def test_mixed_valid_invalid_values(self):
		"""Test with mix of valid and invalid values"""
		data = pd.DataFrame({'km': [10, 20, 'bad', 40, 'worse']})

		result = _convert_columns_to_numeric(data.copy(), ['km'])

		self.assertEqual(result['km'].iloc[0], 10.0)
		self.assertEqual(result['km'].iloc[1], 20.0)
		self.assertTrue(pd.isna(result['km'].iloc[2]))
		self.assertEqual(result['km'].iloc[3], 40.0)
		self.assertTrue(pd.isna(result['km'].iloc[4]))

	def test_empty_dataframe(self):
		"""Test with empty DataFrame"""
		data = pd.DataFrame(columns=['km', 'price'])
		result = _convert_columns_to_numeric(data, ['km', 'price'])

		self.assertEqual(len(result), 0)
		self.assertTrue(pd.api.types.is_numeric_dtype(result['km']))
		self.assertTrue(pd.api.types.is_numeric_dtype(result['price']))

	def test_single_column_conversion(self):
		"""Test converting only one column"""
		data = pd.DataFrame({'km': ['10', '20', '30'], 'price': ['100', '200', '300']})
		result = _convert_columns_to_numeric(data, ['km']) # Do not include 'price'

		self.assertTrue(pd.api.types.is_numeric_dtype(result['km']))
		self.assertFalse(pd.api.types.is_numeric_dtype(result['price']))  # Should remain as string

	def test_nan_values_preserved(self):
		"""Test that existing NaN values are preserved"""
		data = pd.DataFrame({'km': [10, np.nan, 30], 'price': [100, 200, np.nan]})
		result = _convert_columns_to_numeric(data, ['km', 'price'])

		self.assertTrue(pd.isna(result['km'].iloc[1]))
		self.assertTrue(pd.isna(result['price'].iloc[2]))
		self.assertEqual(result['km'].iloc[0], 10.0)

# ==========================================
# _remove_invalid_rows
# ==========================================

class TestRemoveInvalidRows(unittest.TestCase):

	def test_no_invalid_rows(self):
		"""Test when all rows are valid (no NaN values)"""
		data = pd.DataFrame({'km': [10.0, 20.0, 30.0], 'price': [100.0, 200.0, 300.0]})
		result = _remove_invalid_rows(data.copy(), ['km', 'price'])

		pd.testing.assert_frame_equal(result, data)
		self.assertEqual(len(result), 3)

	def test_remove_single_row_with_nan(self):
		"""Test removing a single row with NaN value"""
		data = pd.DataFrame({'km': [10.0, np.nan, 30.0], 'price': [100.0, 200.0, 300.0]})
		result = _remove_invalid_rows(data.copy(), ['km', 'price'])
		self.assertEqual(len(result), 2)
		self.assertEqual(result['km'].tolist(), [10.0, 30.0])
		self.assertEqual(result['price'].tolist(), [100.0, 300.0])

	def test_remove_multiple_rows_with_nan(self):
		"""Test removing multiple rows with NaN values"""
		data = pd.DataFrame({'km': [10.0, np.nan, 30.0, np.nan, 50.0], 'price': [100.0, 200.0, np.nan, 400.0, 500.0]})
		result = _remove_invalid_rows(data.copy(), ['km', 'price'])
		self.assertEqual(len(result), 2)
		self.assertEqual(result['km'].tolist(), [10.0, 50.0])
		self.assertEqual(result['price'].tolist(), [100.0, 500.0])

	def test_all_rows_invalid_raises_error(self):
		"""Test that NoValidDataError is raised when all rows are invalid"""
		data = pd.DataFrame({'km': [np.nan, np.nan, np.nan], 'price': [100.0, 200.0, 300.0]})

		with self.assertRaises(NoValidDataError) as context:
			_remove_invalid_rows(data.copy(), ['km', 'price'])

		self.assertIn("No valid data remaining after cleaning", str(context.exception))

	def test_empty_dataframe_raises_error(self):
		"""Test that NoValidDataError is raised for empty DataFrame"""
		data = pd.DataFrame(columns=['km', 'price'])

		with self.assertRaises(NoValidDataError) as context:
			_remove_invalid_rows(data.copy(), ['km', 'price'])

		self.assertIn("No valid data remaining after cleaning", str(context.exception))

	def test_remove_rows_from_specific_columns_only(self):
		"""Test that only specified columns are checked for NaN"""
		data = pd.DataFrame({'km': [10.0, 20.0, 30.0], 'price': [100.0, np.nan, 300.0], 'year': [2020, 2021, np.nan]})

		# Only check 'km' and 'price', ignore 'year'
		result = _remove_invalid_rows(data.copy(), ['km', 'price'])

		self.assertEqual(len(result), 2)
		self.assertEqual(result['km'].tolist(), [10.0, 30.0])
		# Row with year=NaN should still be present since we only check km and price
		self.assertTrue(pd.isna(result['year'].iloc[1]))


# ==========================================
# clean_data (integration tests)
# ==========================================

class TestCleanData(unittest.TestCase):

	def test_clean_valid_data(self):
		"""Test cleaning data that is already valid"""
		data = pd.DataFrame({'km': [10, 20, 30], 'price': [100, 200, 300]})
		result = clean_data(data.copy(), ['km', 'price'])

		self.assertIsNotNone(result)
		self.assertEqual(len(result), 3)
		self.assertTrue(pd.api.types.is_numeric_dtype(result['km']))
		self.assertTrue(pd.api.types.is_numeric_dtype(result['price']))

	def test_clean_string_numbers(self):
		"""Test cleaning data with string numbers"""
		data = pd.DataFrame({'km': ['10', '20', '30'], 'price': ['100', '200', '300']})
		result = clean_data(data.copy(), ['km', 'price'])

		self.assertIsNotNone(result)
		self.assertEqual(len(result), 3)
		self.assertEqual(result['km'].tolist(), [10.0, 20.0, 30.0])
		self.assertEqual(result['price'].tolist(), [100.0, 200.0, 300.0])

	def test_clean_data_with_invalid_values(self):
		"""Test cleaning data with invalid values that get removed"""
		data = pd.DataFrame({'km': [10, 'invalid', 30, 40], 'price': [100, 200, 'N/A', 400]})
		result = clean_data(data.copy(), ['km', 'price'])

		self.assertIsNotNone(result)
		self.assertEqual(len(result), 2)
		self.assertEqual(result['km'].tolist(), [10.0, 40.0])
		self.assertEqual(result['price'].tolist(), [100.0, 400.0])

	def test_missing_columns_returns_none(self):
		"""Test that missing columns returns None"""
		data = pd.DataFrame({'km': [10, 20, 30]})
		result = clean_data(data.copy(), ['km', 'price'])
		self.assertIsNone(result)

	def test_all_invalid_data_returns_none(self):
		"""Test that all invalid data returns None"""
		data = pd.DataFrame({'km': ['bad', 'worse', 'terrible'], 'price': ['a', 'b', 'c']})
		result = clean_data(data.copy(), ['km', 'price'])
		self.assertIsNone(result)

	def test_mixed_valid_invalid_with_nan(self):
		"""Test cleaning data with mix of valid, invalid, and NaN values"""
		data = pd.DataFrame({
			'km': [10, 'invalid', 30, np.nan, 50],
			'price': [100, 200, np.nan, 400, 'bad']
		})
		result = clean_data(data.copy(), ['km', 'price'])
		self.assertIsNotNone(result)
		# Row 0: valid (10, 100) ✓
		# Row 1: invalid becomes NaN in km, valid price → dropped
		# Row 2: valid km, NaN in price → dropped
		# Row 3: NaN in km, valid price → dropped
		# Row 4: valid km, invalid becomes NaN in price → dropped
		# Only row 0 remains
		self.assertEqual(len(result), 1)
		self.assertEqual(result['km'].tolist(), [10.0])
		self.assertEqual(result['price'].tolist(), [100.0])

	def test_empty_dataframe_returns_none(self):
		"""Test that empty DataFrame returns None"""
		data = pd.DataFrame(columns=['km', 'price'])
		result = clean_data(data.copy(), ['km', 'price'])
		self.assertIsNone(result)

	def test_partial_column_cleaning(self):
		"""Test cleaning only specific columns, leaving others unchanged"""
		data = pd.DataFrame({
			'km': ['10', '20', '30'],
			'price': ['100', '200', '300'],
			'model': ['A', 'B', 'C']
		})

		result = clean_data(data.copy(), ['km', 'price'])

		self.assertIsNotNone(result)
		self.assertEqual(len(result), 3)
		self.assertTrue(pd.api.types.is_numeric_dtype(result['km']))
		self.assertTrue(pd.api.types.is_numeric_dtype(result['price']))
		self.assertFalse(pd.api.types.is_numeric_dtype(result['model']))  # Should remain as string


if __name__ == '__main__':
	unittest.main()
