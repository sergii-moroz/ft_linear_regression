import pandas as pd
from pathlib import Path

def load_data(filepath=None):
	"""Load the dataset from CSV file"""
	if filepath is None:
		filepath = Path(__file__).parent / 'data' / 'ft_linear_regression_data.csv'
	else:
		filepath = Path(filepath)
	print(f"filepath:", filepath)

	try:
		data = pd.read_csv(filepath)
		x = data['km'].values
		y = data['price'].values
		return x, y
	except FileNotFoundError:
		print(f"Error: File '{filepath}' not found")
		return None, None
	except KeyError as e:
		print(f"Error: Column {e} not found in the datase")
		return None, None

def main():
	"""Main function to train the linear regression model"""
	print("=" * 42)
	print("LINEAR REGRESSION - TRAINING")
	print("=" * 42)

	# Load data
	print("Loading data...")
	x, y = load_data()
	if x is None or y is None:
		return
	print("x:", x)
	print("y", y)

if __name__ == "__main__":
	main()
