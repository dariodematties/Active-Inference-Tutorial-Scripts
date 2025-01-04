# Variationlal Free Energy calculation example

import numpy as np
import matplotlib.pyplot as plt

# Set observation, note that this could be set to include more observations.
# For example, it could be set to [0 ,0 , 1] to present a third observation.
# Note that this would require adding a corresponding third row to the Likelihood matrix below
# to specify the probability of a third observation given each hidden state.
# One could similarly add a third hidden state by adding a third column into the  Prior and Likelihood matrices.
True_observation = np.array([0, 1])
print('True observation:', True_observation)

# Generative model parameters
# Prior probability of hidden states
Prior = np.array([0.5, 0.5])    # Prior probability of each hidden state p(s)
print('Prior p(s):', Prior)
# Likelihood matrix, where each row represents the probability of an observation given each hidden state
Likelihood = np.array([[0.8, 0.2], [0.2, 0.8]]) # Likelihood matrix p(o|s) where o is the observation and s is the hidden state 
                                                # Columns represent hidden states, rows represent observations
print('Likelihood p(o|s):', Likelihood)


################## Exact inference (Bayes Theorem) ##################
# Compute the likelihood of the observation given the hidden states
Likelihood_of_observation = np.dot(Likelihood, True_observation)
print('Likelihood of observation given hidden states:', Likelihood_of_observation)

# Compute the joint probability of the observation and hidden states
Joint_probability = np.multiply(Prior, Likelihood_of_observation)  # Joint probability distribution p(o,s) = p(o|s)p(s)
print('Joint probability of observation and hidden states p(o,s) = p(o|s)p(s):', Joint_probability)

# Compute the marginal probability of the observation
Marginal_probability = np.sum(Joint_probability)  # Marginal probability distribution p(o) = sum_s p(o,s)
print('Marginal probability of observation p(o) = sum_s p(o,s):', Marginal_probability)

# Compute the exat posterior probability using Bayes' rule
Posterior = np.divide(Joint_probability, Marginal_probability)  # Posterior probability distribution p(s|o) = p(o,s)/p(o)
print('Exact posterior probability p(s|o) = p(o,s)/p(o):', Posterior)


################## Variational inference (Variational Free Energy) #################
# Note: q(s) = approximated posterior belief: we want to get this as close as possible
# to the true posterior p(s|o) by minimizing the variational free energy

# Different decompositions of the Variational Free Energy (F)
# 1. F=E_q(s)[ln(q(s)/p(o,s))]
# 2. F=E_q(s)[ln(q(s)/p(s))] - E_q(s)[ln(p(o|s))] Complexity-accuracy trade-off version.

# The first term can be interpreted as a complexity term (the KL divergence 
# between prior beliefs p(s) and approximate posterior beliefs q(s)). In 
# other words, how much beliefs have changed after a bew observation.

# The second term (excluding the minus sign) is the accuracy or (including the 
# minus sign) the entropy (= expected surprisal) of observations given 
# approximate posterior beliefs q(s). Written in this way 
# free-energy-minimisation is equivalent to a statistical Occam's razor, 
# where the agent tries to find the most accurate posterior belief that also
# changes its beliefs as little as possible.

# 3. F=E_q(s)[ln(q(s)) - ln(p(s|o)p(o))]
# 4. F=E_q(s)[ln(q(s)/p(s|o))] - E_q(s)[ln(p(o))]

 
# These two versions similarly show F in terms of a difference between
# q(s) and the true posterior p(s|o). Here we focus on #4.

# The first term is the KL divergence between the approximate posterior q(s)  
# and the unknown exact posterior p(s|o), also called the relative entropy. 

# The second term (excluding the minus sign) is the log evidence or (including 
# the minus sign) the surprisal of observations. Note that ln(p(o)) does 
# not depend on q(s), so its expectation value under q(s) is simply ln(p(o)).

# Since this term does not depend on q(s), minimizing free energy means that 
# q(s) comes to approximate p(s|o), which is our unknown, desired quantity.

Initial_approximated_posterior = Prior  # Initial approximated posterior belief
                                        # Set this to the generative model prior

print('Initial approximated posterior belief:', Initial_approximated_posterior)

# Compute the variational free energy (F) using the decomposition #4
# F=E_q(s)[ln(q(s)/p(s|o))] - E_q(s)[ln(p(o))]
Initial_F = Initial_approximated_posterior[0]*(np.log(Initial_approximated_posterior[0]) - np.log(Joint_probability[0])) + Initial_approximated_posterior[1]*(np.log(Initial_approximated_posterior[1]) - np.log(Joint_probability[1]))

print('Initial variational free energy:', Initial_F)

Optimal_approximated_posterior = Posterior  # Optimal approximated posterior belief
                                            # Set this to the exact posterior belief

print('Optimal approximated posterior belief:', Optimal_approximated_posterior)

# Compute the variational free energy (F) using the decomposition #4
# F=E_q(s)[ln(q(s)/p(s|o))] - E_q(s)[ln(p(o))]
Minimized_F = Optimal_approximated_posterior[0]*(np.log(Optimal_approximated_posterior[0]) - np.log(Joint_probability[0])) + Optimal_approximated_posterior[1]*(np.log(Optimal_approximated_posterior[1]) - np.log(Joint_probability[1]))

print('Minimized variational free energy:', Minimized_F)
