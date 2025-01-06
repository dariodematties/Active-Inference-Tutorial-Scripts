# Message passing example


# This script provides two examples of (marginal) message passing, based on
# the steps described in the main text. Each of the two examples (sections)
# need to be run separately. The first example fixes all observed
# variables immediately and does not include variables associated with the
# neural process theory. The second example provides observations
# sequentially and also adds in the neural process theory variables. To
# remind the reader, the message passing steps in the main text are:

# 	1. Initialize the values of the approximate posteriors q(s_(?,?) ) 
#      for all hidden variables (i.e., all edges) in the graph. 
# 	2. Fix the value of observed variables (here, o_?).
# 	3. Choose an edge (V) corresponding to the hidden variable you want to 
#      infer (here, s_(?,?)).
# 	4. Calculate the messages, ?(s_(?,?)), which take on values sent by 
#      each factor node connected to V.
# 	5. Pass a message from each connected factor node N to V (often written 
#      as ?_(N?V)). 
# 	6. Update the approximate posterior represented by V according to the 
#      following rule: q(s_(?,?) )? ? ?(s_(?,?))? ?(s_(?,?)). The arrow 
#      notation here indicates messages from two different factors arriving 
#      at the same edge. 
#       6A. Normalize the product of these messages so that q(s_(?,?) ) 
#           corresponds to a proper probability distribution. 
#       6B. Use this new q(s_(?,?) ) to update the messages sent by 
#           connected factors (i.e., for the next round of message passing).
# 	7. Repeat steps 4-6 sequentially for each edge.
# 	8. Steps 3-7 are then repeated until the difference between updates 
#      converges to some acceptably low value (i.e., resulting in stable 
#      posterior beliefs for all edges). 

# This section carries out marginal message passing on a graph with beliefs
# about states at two time points. In this first example, both observations 
# are fixed from the start (i.e., there are no ts as in full active inference
# models with sequentially presented observations) to provide the simplest
# example possible. We also highlight where each of the message passing
# steps described in the main text are carried out.

# Note that some steps (7 and 8) appear out of order when they involve loops that
# repeat earlier steps

# Specify generative model and initialize variables

import numpy as np
from matplotlib import pyplot as plt

# Functions
# Natural log
def ln(x):
    return np.log(x+1e-16)

# end of functions


# Initialize the priors over states
D = np.array([0.5, 0.5])  # Prior over states

# Initialize the likelihoods of observations given states
A = np.array([[0.9, 0.2], [0.2, 0.9]])  # Likelihood matrix

# Initialize the approximate posterior beliefs over states
B = np.array([[1, 0], [0, 1]])  # Approximate posterior beliefs over states

# Number of time points
T = 2

# Number of iterations of message passing
num_iter = 16

# Initialize values of approximate posterior beliefs over states (Step 1)
Qs = np.zeros((2, T))
for t in range(T):
    Qs[:, t] = np.array([0.5, 0.5])

# Fix the values of observed variables (Step 2)
o = np.zeros((2, T))
for t in range(T):
    o[:, t] = np.array([1, 0])


# Iterate over message passing steps (Steps 3-7)
# for a fixed number of iterations or until convergence (Step 8)
qs = np.zeros((num_iter, 2, T)) # Store the approximate posterior beliefs
for ni in range(num_iter):# Repeat until convergence/fixed number of iterations (Step 8)
    for tau in range(T):# Repeat for the reminder edges (Step 7)
        # Choose an edge corresponding to the hidden
        # variable you want to infer (Step 3)
        V = tau
        q = ln(Qs[:, V])

        # Compute the messages sent by D and B (Step 4)
        # using the posterior computed in step 6B
        if V == 0:  # First time point
            lnD = ln(D)                         # Message 1
            lnBs = ln(np.matmul(B, Qs[:, V+1])) # Message 2
        elif V == T-1:  # Last time point
            lnBs = ln(np.matmul(B.T, Qs[:, V-1])) # Message 1

        # Likelihood of the observation given the state
        lnAo = ln(np.matmul(A, o[:, V]))

        # Steps 5 and 6 (Pass messages and update approximate posterior)
        # Since all terms are in log space, we sum them instead of multiplying
        if V == 0:
            q = 0.5*lnD + 0.5*lnBs + lnAo
        elif V == T-1:
            q = 0.5*lnBs + lnAo

        # Normalize the approximate posterior (Step 6A)
        Qs[:, V] = np.exp(q) / np.sum(np.exp(q))
        qs[ni, :, V] = Qs[:, V] # Store the approximate posterior

Qs # Final approximate posterior beliefs over states

# Show the approximate posterior beliefs over states
print('Approximate posterior beliefs over states:')
print(Qs)

# Plot the approximate posterior beliefs over states
# firing rates (traces)
qs_plot = np.concatenate((np.array([D, D])[None, ...], qs), axis=0)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(qs_plot[:, 0, 0], 'o-', label='State 1')
ax[0].plot(qs_plot[:, 1, 0], 'o-', label='State 2')
ax[0].set_xlabel('Message passing iteration')
ax[0].set_ylabel(f'Approximate posterior belief $q(s_{ 1 })$')
ax[0].set_title('Approximate posterior beliefs over states (Time 1)')
ax[0].legend()
ax[1].plot(qs_plot[:, 0, 1], 'o-', label='State 1')
ax[1].plot(qs_plot[:, 1, 1], 'o-', label='State 2')
ax[1].set_xlabel('Message passing iteration')
ax[1].set_ylabel('Approximate posterior belief $q(s_{ 2 })$')
ax[1].set_title('Approximate posterior beliefs over states (Time 2)')
ax[1].legend()
plt.tight_layout()
plt.show()



