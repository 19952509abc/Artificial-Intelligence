import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt(r'C:\Users\Muhammad Saad\Desktop\AI (Muhammad Saad)(BCS21258)\Lab10\ex1data1.txt', delimiter=',')


# Split data into x_train (Population) and y_train (Profit)
x_train = data[:, 0]
y_train = data[:, 1]


def compute_cost(x, y, w, b):
    m = x.shape[0]  # number of training examples
    total_cost = 0
    
    for i in range(m):
        # Prediction (f_wb)
        f_wb = w * x[i] + b
        # Square of the error
        total_cost += (f_wb - y[i]) ** 2
    
    # Cost calculation
    return total_cost / (2 * m)

def gradient_descent(x, y, w, b, learning_rate, num_iters):
    m = len(x)  # number of training examples
    
    for i in range(num_iters):
        # Initialize gradients
        dw, db = 0, 0
        
        for j in range(m):
            f_wb = w * x[j] + b  # Predicted value
            dw += (f_wb - y[j]) * x[j]  # Gradient w.r.t. w
            db += (f_wb - y[j])  # Gradient w.r.t. b
        
        # Update w and b
        w -= learning_rate * dw / m
        b -= learning_rate * db / m

        # Print the cost every 100 iterations (optional)
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {compute_cost(x, y, w, b)}")
    
    return w, b

# Initial values
initial_w = 0
initial_b = 0
learning_rate = 0.01
num_iters = 1000

# Perform Gradient Descent
w, b = gradient_descent(x_train, y_train, initial_w, initial_b, learning_rate, num_iters)
print(f"Optimized w: {w}, Optimized b: {b}")

# Predicted values
predictions = w * x_train + b

# Plotting
plt.scatter(x_train, y_train, color="red", label="Actual data")
plt.plot(x_train, predictions, label="Linear Regression")

# Annotate the slope (w) and intercept (b) on the graph
plt.text(15, 20, f"Slope (w): {w:.4f}", fontsize=12, color="blue")
plt.text(15, 18, f"Intercept (b): {b:.4f}", fontsize=12, color="blue")

# Labels and title
plt.xlabel("Population (in 10,000s)")
plt.ylabel("Profit (in $10,000s)")
plt.title("Population vs Profit")

# Legend
plt.legend()

# Display the graph
plt.show()