# Example code for simulating state and outcame prediction errors

# Set up the model parameters to calculate the state prediction errors
# This minimizes variational free energy
# (keeps posterior beliefs accurate while keeping them as close as possible to the prior beliefs)

# Import libraries
import numpy as np

def ln(x):
    return np.log(x+np.exp(-16))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

A = np.array([[0.8, 0.4], [0.2, 0.6]])  # Likelihood matrix

B_t1 = np.array([[0.9, 0.2], [0.1, 0.8]])  # Transition prior matrix from previous time step
B_t2 = np.array([[0.2, 0.3], [0.8, 0.7]])  # Transition prior matrix from current time step

o = np.array([1.0, 0.0]).T  # Observed variable

s_pi_tau = np.array([0.5, 0.5]).T  # Prior beliefs over states
                                   # Note that we use the same value for  s_pi_tau-1
                                   # s_pi_tau and s_pi_tau+1
                                   # but this need not be the case

s_pi_tau_minus_1 = np.array([0.5, 0.5]).T  # Prior beliefs over states at time step t-1
s_pi_tau_plus_1 = np.array([0.5, 0.5]).T  # Prior beliefs over states at time step t+1

v_0 = ln(s_pi_tau)  # Initial value of the hidden variable (depolarization term)

B_t2_cross_intermediate = B_t2.T

B_t2_cross = softmax(B_t2_cross_intermediate)

# Calculate the state prediction error

state_prediction_error = 1/2 * (ln(B_t1@s_pi_tau_minus_1) + ln(B_t2_cross@s_pi_tau_plus_1)) + ln(A.T@o) - ln(s_pi_tau) # State prediction error

v = v_0 + state_prediction_error  # Update the depolarization variable

s = softmax(v)  # Normalize using a softmax function to find the posterior

print(f'Prior Distribution over states: {s_pi_tau}')
print(f'State prediction error: {state_prediction_error}')
print(f'Depolarization variable: {v}')
print(f'Posterior Distribution over states: {s}')


# Set upt the model to calculate the outcome prediction error
# This minimizes expected free energy
# (maximizes reward and information gain)

A = np.array([[0.9, 0.1], [0.1, 0.9]])  # Likelihood matrix

S1 = np.array([0.9, 0.1]).T  # States under policy 1
S2 = np.array([0.5, 0.5]).T  # States under policy 2

C = np.array([1, 0]).T  # Preferred outcomes

o_1 = A@S1  # Predicted outcomes under policy 1
o_2 = A@S2  # Predicted outcomes under policy 2

risk_1 = np.dot(o_1, ln(o_1) - ln(C))  # Risk under policy 1
risk_2 = np.dot(o_2, ln(o_2) - ln(C))  # Risk under policy 2

print(f'Risk under policy 1: {risk_1}')
print(f'Risk under policy 2: {risk_2}')

# Calculate the ambiguity (information-seeking) term under the two policies

A = np.array([[0.4, 0.2], [0.6, 0.8]])  # Likelihood matrix

s1 = np.array([0.9, 0.1]).T  # States under policy 1
s2 = np.array([0.1, 0.9]).T  # States under policy 2

ambiguity_1 = -np.dot(np.diag(A.T@ln(A)), s1)  # Ambiguity under policy 1
ambiguity_2 = -np.dot(np.diag(A.T@ln(A)), s2)  # Ambiguity under policy 2

print(f'Ambiguity under policy 1: {ambiguity_1}')
print(f'Ambiguity under policy 2: {ambiguity_2}')


