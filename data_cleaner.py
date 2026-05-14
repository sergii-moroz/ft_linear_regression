import pandas as pd


class MissingColumnsError(Exception):
	"""Raised when required columns are missing from the dataset"""
	pass


class NoValidDataError(Exception):
	"""Raised when no valid data remains after cleaning"""
	pass


def _check_required_columns(data, required_columns):
	"""Check if all required columns exist in the dataset

	Args:
		data: pandas DataFrame
		required_columns: list of column names to check

	Raises:
		MissingColumnsError: If any required columns are missing
	"""
	missing_columns = [col for col in required_columns if col not in data.columns]
	if missing_columns:
		raise MissingColumnsError(f"Missing required column(s): {missing_columns}")

def _convert_columns_to_numeric(data, columns):
	"""Convert specified columns to numeric, reporting invalid values

	Args:
		data: pandas DataFrame
		columns: list of column names to convert

	Returns:
		pandas DataFrame with converted columns
	"""
	for column in columns:
		invalid_mask = pd.to_numeric(data[column], errors='coerce').isna() & data[column].notna()
		if invalid_mask.any():
			invalid_indices = data[invalid_mask].index.tolist()
			print(f"Warning: Column '{column}' has {invalid_mask.sum()} non-numeric value(s) at row(s): {invalid_indices}")
		data[column] = pd.to_numeric(data[column], errors='coerce')
	return data

def _remove_invalid_rows(data, columns):
	"""Remove rows with NaN values in specified columns

	Args:
		data: pandas DataFrame
		columns: list of column names to check for NaN

	Returns:
		pandas DataFrame with invalid rows removed

	Raises:
		NoValidDataError: If no valid data remains after cleaning
	"""
	original_len = len(data)
	data = data.dropna(subset=columns)
	dropped = original_len - len(data)

	if dropped > 0:
		print(f"Dropped {dropped} row(s) with invalid or missing values")

	if len(data) == 0:
		raise NoValidDataError("No valid data remaining after cleaning")

	return data

def clean_data(data, required_columns):
	"""Clean and validate the dataset by converting to numeric and dropping invalid rows

	Args:
		data: pandas DataFrame
		required_columns: list of column names that must be numeric

	Returns:
		pandas DataFrame with validated numeric columns, or None if validation fails
	"""
	try:
		_check_required_columns(data, required_columns)
	except MissingColumnsError as e:
		print(f"Error: {e}")
		return None

	data = _convert_columns_to_numeric(data, required_columns)

	try:
		data = _remove_invalid_rows(data, required_columns)
	except NoValidDataError as e:
		print(f"Error: {e}")
		return None

	return data
