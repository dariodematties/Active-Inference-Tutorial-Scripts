# Example code for simulated expected free energy (EFE) precision (beta/gamma) updates

# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# This script will reproduce the simulation results in Figure 9

# Here you can set the number of policies and the distributions that
# contribute to prior and posterior policy precision

# Set a fixed-form prior distribution over policies (habits)
E = np.array([1, 1, 1, 1, 1])

# Set an example expected free energy distribution over policies
G = np.array([12.505, 9.51, 12.5034, 12.505, 12.505]) 

# Set an example of variational free energy distribution
# over policies after a new observation
F = np.array([17.0207, 1.7321, 1.7321, 17.0387, 17.0387])

# Starting expected free energy precision value
gamma_0 = 1
# Initial expected free energy precision to be updated
gamma = gamma_0
# Initial prior on expected free energy precision
beta_prior = 1/gamma
# Initial posterior on expected free energy precision
beta_posterior = beta_prior
# Step size parameter (promotes stable convergence)
psi = 2

# Number of variational updates (16)
num_updates = 16
gamma_dopamine = np.zeros((num_updates))
policies_neural = np.zeros((F.shape[0], num_updates))
for ni in range(num_updates):

    # Compute the prior and posterior over policies

    # Compute the prior over policies
    pi_0 = np.exp(np.log(E) - gamma*G) / np.sum(np.exp(np.log(E) - gamma*G))

    # Compute the posterior over policies
    pi = np.exp(np.log(E) - gamma*G - F) / np.sum(np.exp(np.log(E) - gamma*G - F))

    # Compute the expected free energy precision update
    G_error = np.dot((pi - pi_0), -G)

    # Compute the change in beta: gradient of F with respect to gamma
    # ( recal gamma = 1/beta )
    beta_update = beta_posterior - beta_prior + G_error

    # Update the posterior precision estimate (with a step size of psi = 2
    # which reduces the magnitude of each update and can promote stable convergence)
    beta_posterior = beta_posterior - beta_update / psi

    # Update expected free energy precision
    gamma = 1/beta_posterior

    # Now simulate dopamine responses
    n = ni

    # Simulated neural encoding of precision (beta_posterior⁻¹)
    # at each iteration of the variational update
    gamma_dopamine[n] = gamma

    # Neural encoding of posterior over policies
    # at each iteration of the variational updating
    policies_neural[:, n] = pi



# Show the results
print('Final policy prior:', pi_0)
print('Final policy posterior:', pi)
print('Final Policy Difference Vector:', pi - pi_0)
print('Negative expected free energy:', -G)
print('Prior G Precision (Prior Gamma):', gamma_0)
print('Posterior G Precision (Posterior):', gamma)

# Include prior value
gamma_dopamine_plot = [gamma_i.item() for gamma_i in gamma_dopamine]
gamma_dopamine_plot = [gamma_0, gamma_0, gamma_0] + gamma_dopamine_plot
print('Dopamine Response:', gamma_dopamine_plot)

# Plot the results
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(gamma_dopamine_plot, 'o-')
ax[0].set_ylim(np.min(gamma_dopamine_plot) - 0.01, np.max(gamma_dopamine_plot) + 0.01)
ax[0].set_title('Expected Free Energy Precision (Tonic Dopamine Response)')
ax[0].set_xlabel('Variational Update')
ax[0].set_ylabel('Gamma')

ax[1].plot(np.gradient(gamma_dopamine_plot), 'o-')
ax[1].set_ylim(np.min(np.gradient(gamma_dopamine_plot)) - 0.01, np.max(np.gradient(gamma_dopamine_plot)) + 0.01)
ax[1].set_title('Rate of Change in Precision (Phasic Dopamine Response)')
ax[1].set_xlabel('Variational Update')
ax[1].set_ylabel('Gamma gradient')

plt.tight_layout()
plt.show()

