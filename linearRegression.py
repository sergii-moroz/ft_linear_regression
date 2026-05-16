import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
	"""Linear Regression model using gradient descent"""

	def __init__(self, learning_rate: float = 0.01, iterations: int = 1000) -> None:
		self.learning_rate = learning_rate
		self.iterations = iterations
		self.a: float = 0.0
		self.b: float = 0.0
		self.cost_history: list[float] = []

		self.x_min = 0
		self.x_max = 1
		self.y_min = 0
		self.y_max = 1

	def normalize(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		"""Normalize fetures using min-max normalization"""
		self.x_min = np.min(x)
		self.x_max = np.max(x)
		self.y_min = np.min(y)
		self.y_max = np.max(y)

		x_range = self.x_max - self.x_min
		y_range = self.y_max - self.y_min

		if x_range == 0:
			x_norm = np.zeros_like(x)
		else:
			x_norm = (x - self.x_min) / x_range

		if y_range == 0:
			y_norm = np.zeros_like(y)
		else:
			y_norm = (y - self.y_min) / y_range

		return x_norm, y_norm

	def denormalize_parameters(self) -> tuple[float, float]:
		"""Convert normalized parameters back to original scale"""
		x_range = self.x_max - self.x_min
		y_range = self.y_max - self.y_min

		if x_range == 0:
			a = 0.0
			b = self.y_min + y_range * self.b
		else:
			a = self.a * y_range / x_range
			b = self.y_min + y_range * (self.b - self.a * self.x_min / x_range)

		return a, b

	def predict(self, x: np.ndarray) -> np.ndarray:
		"""Make predictions using the hypothesis function f(x) = ax + b"""
		return self.a * x + self.b

	def compute_cost(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
		"""Compute the cost function - Mean Squared Error (MSE)"""
		m = len(x)
		predictions = self.predict(x)
		cost = 1 / (2 * m) * np.sum((predictions - y)**2)
		return cost

	def gradient_descent(self, x: np.ndarray, y: np.ndarray):
		"""Perform gradient descent to learn 'a' and 'b' parameters"""
		m = len(x)

		for i in range(self.iterations):
			# Calculate predictions
			predictions = self.predict(x)

			# Calculate errors
			errors = predictions - y

			# Calculate gradients
			gradient_a = 1 / m * np.sum(errors * x)
			gradient_b = 1 / m * np.sum(errors)

			# Update parameters
			self.a = self.a - self.learning_rate * gradient_a
			self.b = self.b - self.learning_rate * gradient_b

			# Calculate and strore cost
			cost = self.compute_cost(x, y)
			self.cost_history.append(cost)

			# Print progress every 100 iterations
			if (i + 1) % 1000 == 0:
				print(f"Iteration {i + 1}/{self.iterations} - Cost: {cost:.6f}")

	def fit(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
		"""Train the model on the given data"""
		print("Normalizing data...")
		x_norm, y_norm = self.normalize(x, y)

		print(f"Training with learning rate: {self.learning_rate}")
		print(f"Number of iterations: {self.iterations}")
		print(f"Number of training examples: {len(x)}")

		self.gradient_descent(x_norm, y_norm)

		print(f"Training completed!")
		print(f"Final cost: {self.cost_history[-1]:.6f}")
		print(f"Normalized parameters:")
		print(f"  a = {self.a:.6f}")
		print(f"  b =  {self.b:.6f}")

		# Denormalize parameters
		a, b = self.denormalize_parameters()
		print(f"Denormalized parameters:")
		print(f"  a = {a:.6f}")
		print(f"  b = {b:.6f}")

		return a, b

	def plot_cost_history(self) -> None:
		"""Plot the cost function over iterations"""
		iterations = range(1, len(self.cost_history) + 1)
		costs = self.cost_history

		# Check if we can use log scale (all costs must be positive)
		has_positive_costs = all(cost > 0 for cost in costs)

		if has_positive_costs:
			fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
		else:
			fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

		# First subplot - Original scale
		ax1.plot(iterations, costs, 'b-', linewidth=2)
		ax1.set_xlabel('Iteration')
		ax1.set_ylabel('Cost (MSE)')
		ax1.set_title('Cost Function - Original Scale')
		ax1.grid(True, alpha=0.3)

		# Second subplot - Log-log scale (only if costs are positive)
		if has_positive_costs:
			ax2.loglog(iterations, costs, "r-", linewidth=2)
			ax2.set_xlabel('Iteration')
			ax2.set_ylabel('Cost (MSE)')
			ax2.set_title('Cost Function - Log-Log Scale')
			ax2.grid(True, alpha=0.3, which="both")

		plt.tight_layout()
		plt.show()
