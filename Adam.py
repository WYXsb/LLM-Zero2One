import numpy as np

class AdamOptimizer:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Adam Optimizer implementation

        :param params: model parameters (weights and biases)
        :param lr: learning rate
        :param beta1: coefficient for first moment estimate (default 0.9)
        :param beta2: coefficient for second moment estimate (default 0.999)
        :param epsilon: small constant to prevent division by zero (default 1e-8)
        """
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {key: np.zeros_like(value) for key, value in params.items()}  # first moment vector
        self.v = {key: np.zeros_like(value) for key, value in params.items()}  # second moment vector
        self.t = 0  # timestep

    def update(self, grads):
        """
        Update the parameters using Adam optimizer

        :param grads: gradients of the loss w.r.t parameters
        """
        self.t += 1
        # For each parameter in the model, update its value
        for key in self.params:
            g = grads[key]  # gradient of the parameter

            # Update the first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            # Update the second moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g ** 2)

            # Apply bias correction
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # Update the parameter
            self.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def get_params(self):
        """Return model parameters."""
        return self.params

# Example usage

# Initialize model parameters (weights)
params = {
    'W1': np.array([0.5, 0.5]),  # Simple 2D weight vector
    'b1': np.array([0.1]),        # Bias term
}

# Define the Adam optimizer
optimizer = AdamOptimizer(params, lr=0.001)

# Simulate gradient computation (for illustration purposes, these would come from backpropagation in a real model)
grads = {
    'W1': np.array([0.1, -0.2]),  # Gradient for W1
    'b1': np.array([0.05]),        # Gradient for b1
}

# Perform a few optimization steps
for step in range(1, 11):  # Running 10 steps of optimization
    print(f"Step {step}:")
    print(f"Before update: W1 = {params['W1']}, b1 = {params['b1']}")

    # Update parameters with Adam
    optimizer.update(grads)

    # Get the updated parameters
    updated_params = optimizer.get_params()

    print(f"After update: W1 = {updated_params['W1']}, b1 = {updated_params['b1']}")
    print('-' * 50)
