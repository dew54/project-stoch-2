import numpy as np

# Define the transition matrix for the Markov chain
transition_matrix = np.array([[0.7, 0.2, 0.1],
                             [0.3, 0.5, 0.2],
                             [0.1, 0.3, 0.6]])

# Define the initial state distribution
initial_state = np.array([0.3, 0.4, 0.3])

# Define the 3D vectorial initial state
initial_state_vector = np.array([[1.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0],
                                 [0.0, 0.0, 1.0]])

# Define the number of steps to simulate
num_steps = 10

# Simulate the Markov chain
state_vector = np.zeros((num_steps, 3, 3))
state_vector[0] = initial_state_vector * initial_state[:, np.newaxis, np.newaxis]
for t in range(1, num_steps):
    state_vector[t] = np.einsum('ij,kjl->kil', transition_matrix, state_vector[t-1])

# Print the simulated states
print('Simulated states:\n', state_vector)