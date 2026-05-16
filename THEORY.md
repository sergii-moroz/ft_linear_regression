# Linear Regression - Theoretical Explanation

## The Mathematical Model

### The Hypothesis Function

For simple linear regression with one feature, the hypothesis function is:

$$f(x) = a \cdot x + b$$

Where:
- $f(x)$ is redicted value (estimated price)
- $x$ is input feature (mileage in km)
- $a$ is slope (how much the price changes per km)
- $b$ is y-intercept (the predicted price when mileage is 0)

### The Goal

The goal is to find the optimal values of $a$ and $b$ that best fit trainings datga. "Best fit" means the line that minimizes the difference between predicted prices and actual prices.

## Gradient Descent Algorithm

### Overview

Gradient descent is an iterative optimization algorithm used to find the values of parameters (here: $a$ and $b$) that minimize the *cost function*.

### Problem Setup

We start with a simple dataset. If we fit a line to the data, then for any given $x$, we can predict $f(x)$. Our goal is to find the optimal slope $a$ and intercept $b$ so that the line fits the data well.

We define the line equestion as:

$$f(x) = a · x + b$$

### Initial Guess

We begin by picking random values for $a$ and $b$. This initial guess gives Gradient Descent a starting point to imporve upon.

## Cost Function (Mean Squared Error)

To measure how well line fits the data, the **Mean Squared Error (MSE)** cost function would be used:

- Residual for a single data point $i$:

$$residual^i = f(x^i) - y^i$$


Where $(x^i, y^i)$ is the $i$-th point of the training example.

- To make all residuals positive and emphasize large errors, we square each residual:

$$\left(f(x^i) - y^i\right)^2$$

- Then we average over all $m$ trainings points:

$$J(a, b) =  \frac{1}{2m} \cdot \sum_{i=1}^m \left(f(x^i) - y^i\right)^2$$


The factor $1/2$ is often added to simplify the derivative calculation (it cancels out later).

- $J$ is cost function value
- $m$ is number of trainings examples
- $x^i$ is $i$-th input value (milleage)
- $y^i$ is $i$-th actual output value (actual price)
- $f(x^i)$ is predicted value for *i*-th input value
- $\sum$ is sum over all trainings examples

Substituting $f(x^i) = a \cdot x^i + b$:

$$J(a,b) =  \frac{1}{2m} \cdot \sum_{i=1}^m\left(a \cdot x^i + b - y^i\right)^2$$


This cost function $J(a,b)$ is a quadratic parabaloid with:
- two unknown parameters: $a$ and $b$
- known fixed values: $x^i$, $y^i$

For any pair $(a,b)$, $J(a,b)$ gives a cost value. There exists an optimal $(a,b)$ that minimizes this cost.

## Gradient Descent Update Rule

Gradient Descent works by repeatedly updating $a$ and $b$ in the direciton opposite to the gradient.

We assume that the step size is proportional to the partial derivative of the cost function at the current point.

Update rules:
$$a_{new} = a_{old} + \lambda \cdot \frac{\partial J}{\partial a}$$
$$a_{new} = a_{old} + \lambda \cdot \frac{\partial J}{\partial a}$$

Here:
- $\lambda$ is the learning rate.
- $\frac{\partial J}{\partial a}$ and $\frac{\partial J}{\partial b}$ are the partial derivatives (gradients).

>Note: When far from the optimum, the gradient is large → large steps. Near the optimum, the gradient is small → small steps.

### Partial Derivatives of the Cost Function

We compute the gradient of $J(a,b)$ with repect to $a$ and $b$.

#### Derivative with respect to $a$

$$\frac{\partial J}{\partial a} = \frac{1}{2m} \cdot 2 \cdot \sum_{i=1}^m(a \cdot x^i + b - y^i) \cdot x^i$$
$$\frac{\partial J}{\partial a} = \frac{1}{m} \cdot \sum_{i=1}^m(a \cdot x^i + b - y^i) \cdot x^i$$

#### Derivative with respect to $b$

$$\frac{\partial J}{\partial b} = \frac{1}{2m} \cdot 2 \cdot \sum_{i=1}^m(a \cdot x^i + b - y^i)$$
$$\frac{\partial J}{\partial b} = \frac{1}{m} \cdot \sum_{i=1}^m(a \cdot x^i + b - y^i)$$

## Final Gradient Descent Steps

Substitute these deriatives into the update rules:
$$a_{new} = a_{old} + \lambda \cdot \left[\frac{1}{m} \cdot \sum_{i=1}^m(a \cdot x^i + b - y^i) \cdot x^i\right]$$
$$b_{new} = b_{old} + \lambda \cdot \left[\frac{1}{m} \cdot \sum_{i=1}^m(a \cdot x^i + b - y^i)\right] $$

We reapeat these updates until the cost function converges.

### Learning Rate $\lambda$

The Learning rate controls how big each step is:
- **Too small**: Training is very slow, takes many iterations
- **Too large**: May overshoot the minimum, fail to converge, or diverge

Typical values: $0.001$ to $0.1$

## Feature Normalization

When features have very different scales (like mileage: 0-240000 vs price: 3650-8290), gradient descent can be slow and unstable.

### Why Normalize?

- Makes gradient descent converge faster
- Prevents numerical instability
- Ensures all features contribute equally

### Min-Max Normalization

$$\overline{x} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

$$\overline{y} = \frac{y - y_{min}}{y_{max} - y_{min}}$$
This scales values to the range $[0, 1]$

Then you fit a line on the normalized data:

$$\overline{y} = \overline{a} \cdot \overline{x} + \overline{b}$$

### Denormalization

After training on normalized data, we must denormalize the parameters to make predicitons on original scale data.

- Let's start with normalized equestion

$$\overline{y} = \overline{a} \cdot \overline{x} + \overline{b}$$

- Substitute the normalization formulas

$$\frac{y - y_{min}}{y_{max} - y_{min}} = \overline{a} \cdot \frac{x - x_{min}}{x_{max} - x_{min}} + \overline{b}$$

- Multiply both sides by $y_{max} - y_{min}$

$$y - y_{min} = \overline{a} \cdot \frac{x - x_{min}}{x_{max} - x_{min}} \cdot (y_{max} - y_{min}) + \overline{b} \cdot (y_{max} - y_{min})$$

- Let's expand the term with $x$

$$y - y_{min} = \overline{a} \cdot \frac{y_{max} - y_{min}}{x_{max} - x_{min}} \cdot x - \overline{a} \cdot \frac{y_{max} - y_{min}}{x_{max} - x_{min}} \cdot x_{min} + \overline{b} \cdot (y_{max} - y_{min})$$

- Isolate $y$

$$y = \overline{a} \cdot \frac{y_{max} - y_{min}}{x_{max} - x_{min}} \cdot x - \overline{a} \cdot \frac{y_{max} - y_{min}}{x_{max} - x_{min}} \cdot x_{min} + \overline{b} \cdot (y_{max} - y_{min}) + y_{min}$$

### Original slope $a$

$$a = \overline{a} \cdot \frac{y_{max} - y_{min}}{x_{max} - x_{min}}$$

### Original intercept $b$

$$b = y_{min} - \overline{a} \cdot \frac{y_{max} - y_{min}}{x_{max} - x_{min}} \cdot x_{min} + \overline{b} \cdot (y_{max} - y_{min})$$

or more compactly:

$$b = y_{min} + (y_{max} - y_{min}) \cdot \left(\overline{b} - \overline{a} \cdot \frac{x_{min}}{x_{max} - x_{min}}\right) $$

