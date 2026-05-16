import pandas as pd
from pathlib import Path
from data_cleaner import clean_data
from linearRegression import LinearRegression

def get_data_path(filepath=None):
	"""Get the data file path, handling OS-specific separators"""
	if filepath is None:
		return Path(__file__).parent / 'data' / 'ft_linear_regression_data.csv'
	return Path(filepath)

def load_data(filepath=None):
	"""Load the dataset from CSV file"""
	filepath = get_data_path(filepath)

	try:
		data = pd.read_csv(filepath)

		# Clean and validate data
		required_columns = ['km', 'price']
		data = clean_data(data, required_columns)

		if data is None:
			return None, None

		x = data['km'].values
		y = data['price'].values
		return x, y
	except FileNotFoundError:
		print(f"Error: File '{filepath}' not found")
		return None, None
	except Exception as e:
		print(f"Error loading data: {e}")
		return None, None

def main():
	"""Main function to train the linear regression model"""
	print("=" * 42)
	print("LINEAR REGRESSION - TRAINING")
	print("=" * 42)

	# Load data
	print("Loading data...")
	x, y = load_data()#"data/test.csv")
	if x is None or y is None:
		return
	print("x:", x)
	print("y", y)

	# Create and train model
	model = LinearRegression(learning_rate=0.01, iterations=20000)
	a, b = model.fit(x, y)

	print("=" * 42)
	print("TRAINING COMPLETE")
	print("=" * 42)
	print(f"Final model: price = {a:.6f} * mileage + {b:.2f}")

	# Plot cost history
	print(f"Generating cost history plot")
	model.plot_cost_history()

if __name__ == "__main__":
	main()
