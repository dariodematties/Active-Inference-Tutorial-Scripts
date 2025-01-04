# This is the second example of message passing in the script. This example
# includes the Sequential observations and simulation of firing rates and ERPs

# This script performs state estimation using the message passing 
# algorithm introduced in Parr, Markovic, Kiebel, & Friston (2019).
# This script can be thought of as the full message passing solution to 
# problem 2 in the pencil and paper exercises. It also generates
# simulated firing rates and ERPs in the same manner as those shown in
# figs. 8, 10, 11, 14, 15, and 16. Unlike example 1, observations are
# presented sequentially (i.e., two ts and two taus).

# First, we define the generative model and initialize the variables.

import numpy as np
from matplotlib import pyplot as plt

# Functions
# Natural log
def ln(x):
    return np.log(x+1e-16)

# end of functions

# Priors over states
D = np.array([0.5, 0.5])  # Prior over states

# Likelihoods of observations given states
A = np.array([[0.9, 0.2], [0.2, 0.9]])  # Likelihood matrix

# Transitions between states
B = np.array([[1, 0], [0, 1]])  # Transition matrix

# Number of time points
T = 2

# Number of iterations of message passing
num_iter = 16

# Initialize values of approximate posterior beliefs over states (Step 1)
Qs = np.zeros((D.shape[0], T))
for t in range(T):
    Qs[:, t] = np.array([0.5, 0.5])

# Fix the values of observed variables (Step 2)
o = np.zeros((D.shape[0], T, T))
o[:, 0, 0] = np.array([1, 0])
o[:, 0, 1] = np.array([0, 0])
o[:, 1, 0] = np.array([1, 0])
o[:, 1, 1] = np.array([1, 0])

# Iterate over message passing steps

epsilon = np.zeros((D.shape[0], num_iter, T, T))
xn = np.zeros((num_iter, D.shape[0], T, T))
for t in range(T):
    for ni in range(num_iter): # ( Step 8 loop of VMP )
        for tau in range(T): # ( Step 7 loop of VMP )
            # Choose an edge corresponding to the
            # hidden variable you want to infer (Step 3)
            v = ln(Qs[:, t])

            # Compute the messages sent by D and B (Step 4)
            # using the posterior computed in step 6B
            if tau == 0: # First time point
                lnD = ln(D) # past Message 1
                lnBs = ln(np.matmul(B, Qs[:, tau+1])) # future Message 2
            elif tau == T-1: # Last time point
                lnBs = ln(np.matmul(B, Qs[:, tau-1])) # no contribution from future,
                                                      # ( only Message 1 )

            # Likelihood of the observation given the state # Message 3
            lnAo = ln(np.matmul(A.T, o[:, t, tau]))

            # compute state prediction error: equation 24
            if tau == 0:
                epsilon[:, ni, t, tau] = 0.5*lnD + 0.5*lnBs + lnAo - v
            elif tau == T-1:
                epsilon[:, ni, t, tau] = 0.5*lnBs + lnAo - v

            # (Step 6 of VMP)
            # update depolarization variable: equation 25
            v = v + epsilon[:, ni, t, tau]

            # normalize using a softmax function to find the posterior
            # equation 26 (Step 6A of VMP)
            Qs[:, tau] = np.exp(v) / np.sum(np.exp(v))
            # store Qs for firing rate plots
            xn[ni, :, tau, t] = Qs[:, tau]

# Display the final approximate posterior beliefs over states
print('Approximate posterior beliefs over states:')
print(Qs)


# Plots

# Get the firing rates into usable format
num_states = D.shape[0]
num_epochs = T
time_tau = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])

firing_rate = np.zeros((num_iter, num_states, num_epochs*num_epochs)) # Firing rates
ERP = np.zeros((num_iter, num_states, num_epochs*num_epochs)) # Evoked response potentials
for t_tau in range(time_tau.shape[1]):
    for epoch in range(num_epochs):
        # firing rate
        firing_rate[:, epoch, t_tau] = xn[:, time_tau[0, t_tau], time_tau[1, t_tau], epoch]
        ERP[:, epoch, t_tau] = np.gradient(firing_rate[:, epoch, t_tau])


firing_rate = firing_rate.transpose((2, 1, 0))
ERP = ERP.transpose((2, 1, 0))
firing_rate = firing_rate.reshape((-1, num_iter*num_states))
ERP = ERP.reshape((-1, num_iter*num_states))

# add prior for starting values
firing_rate = np.concatenate((np.array([D, D]).reshape(-1, 1), firing_rate), axis=1)
# add 0 for starting values
ERP = np.concatenate((0*np.array([D, D]).reshape(-1, 1), ERP), axis=1)

# firing rates
plt.figure(figsize=(10, 6))
plt.imshow(
        64*(1-firing_rate),
        cmap='gray',
        aspect='auto',
        extent=[0, (num_iter*T), 0, (num_states*num_epochs)]
        )
plt.xlabel('Message passing iteration')
plt.ylabel('Firing rate')
plt.title('Firing rates (Darker = higher firing rate)')
plt.show(block=False)


# Firing rates (traces)
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
domine = [i for i in range(firing_rate[0, :].shape[-1])]
ax[0,0].plot(domine, firing_rate[0, :], 'x-',
                   firing_rate[1, :], 'o-', label='t tau 0 0 and 0 1')
ax[0,0].set_xlabel('Message passing iteration')
ax[0,0].set_ylabel(f'Firing rate')
ax[0,0].set_title('Firing rates (t_tau 0 and 1)')
ax[0,0].legend()
ax[0,1].plot(domine, firing_rate[2, :], 'x-',
                   firing_rate[3, :], 'o-', label='t tau 1 0 and 1 1')
ax[0,1].set_xlabel('Message passing iteration')
ax[0,1].set_ylabel(f'Firing rate')
ax[0,1].set_title('Firing rates (t_tau 2 and 3)')
ax[0,1].legend()
domine = [i for i in range(ERP[0, :].shape[-1])]
ax[1,0].plot(domine, ERP[0, :], 'x-',
                   ERP[1, :], 'o-', label='t tau 0 0 and 0 1')
ax[1,0].set_xlabel('Message passing iteration')
ax[1,0].set_ylabel(f'Firing rate')
ax[1,0].set_title('ERPs (t_tau 0 and 1)')
ax[1,0].legend()
ax[1,1].plot(domine, ERP[2, :], 'x-',
                   ERP[3, :], 'o-', label='t tau 1 0 and 1 1')
ax[1,1].set_xlabel('Message passing iteration')
ax[1,1].set_ylabel(f'Firing rate')
ax[1,1].set_title('ERPs (t_tau 2 and 3)')
ax[1,1].legend()
plt.tight_layout()
plt.show()



