# Step by step introduction to building and using active inference models

import numpy as np

# Simulation options after model building below:

# If Sim = 1, simulate single trial. This will reproduce fig. 8. (Although
            # note that, for this and the following simulations, results 
            # will vary each time due to random sampling)

# If Sim = 2, simulate multiple trials where the left context is active 
            # (D{1} = [1 0]'). This will reproduce fig. 10.
             
# If Sim = 3, simulate reversal learning, where the left context is active 
            # (D{1} = [1 0]') in early trials and then reverses in later 
            # trials (D{1} = [0 1]'). This will reproduce fig. 11.
            
# If Sim = 4, run parameter estimation on simulated data with reversal
            # learning. This will reproduce the top panel of fig. 17.
            
# If Sim = 5, run parameter estimation on simulated data with reversal
            # learning from multiple participants under different models
            # (i.e., different parameter values) and perform model comparison. 
            # This will reproduce the bottom panel of fig. 17. This option
            # will also save two structures that include results of model
            # comparison, model fitting, parameter recoverability analyses,
            # and inputs needed for group (PEB) analyses.
            # PEB stands for parametric empirical Bayes.
            # Parametric Empirical Bayes (PEB) is a statistical method
            # where the prior distribution in a Bayesian analysis is 
            # estimated from the data itself, but with the added constraint 
            # of assuming a specific parametric form for that prior distribution, 
            # allowing for a more structured approach to estimating unknown 
            # parameters compared to standard empirical Bayes methods.


            
rs1 = 4  # Risk-seeking parameter (set to the variable rs below) 
         # To reproduce fig. 8, use values of 4 or 8 (with Sim = 1)
         # To reproduce fig. 10, use values of 3 or 4 (with Sim = 2)
         # To reproduce fig. 11, use values of 3 or 4 (with Sim = 3)
         # This will have no effect on Sim = 4 or Sim = 5

Sim = 1

# When Sim = 5, if PEB = 1 the script will run simulated group-level
# (Parametric Empirical Bayes) analyses.

PEB = 0; # Note: GCM_2 and GCM_3 (the inputs to PEB; see below) are saved 
         # after running Sim = 5 to avoid needing to re-run it each time 
         # you want to use PEB (i.e., because Sim = 5 takes a long time). 
         # After running Sim = 5 once, you can simply load GCM_2 and GCM_3 and 
         # run the PEB section separately if you want to come back to it later.

# You can also run the sections separately after building the model by
# simply clicking into that section and clicking 'Run Section' above

## 1. Set up the generative model structure

# Number of time points or 'epochs' within a trial: T
# =========================================================================

# Here, we specify 3 time points (T), in which the agent 1) starts in a 'Start'
# state, 2) first moves to either a 'Hint' state or a 'Choose Left' or 'Choose
# Right' slot machine state, and 3) either moves from the Hint state to one
# of the choice states or moves from one of the choice states back to the
# Start state.

T = 3

# Priors about initial states: D and d
# =========================================================================

#--------------------------------------------------------------------------
# Specify prior probabilities about initial states in the generative 
# process (D)
# Note: By default, these will also be the priors for the generative model
#--------------------------------------------------------------------------
D = {}

# For the 'context' state factor, we can specify that the 'left better' context 
# (i.e., where the left slot machine is more likely to win) is the true context:

D[1] = np.array([1, 0]).T  # {'left better','right better'}

# For the 'behavior' state factor, we can specify that the agent always
# begins a trial in the 'start' state (i.e., before choosing to either pick
# a slot machine or first ask for a hint:

D[2] = np.array([1, 0, 0, 0]).T  # {'start','hint','choose-left','choose-right'}

# print(f'\nPrior probabilities over initial states: {D}')
#--------------------------------------------------------------------------
# Specify prior beliefs about initial states in the generative model (d)
# Note: This is optional, and will simulate learning priors over states 
# if specified.
#--------------------------------------------------------------------------
d = {}

# Note that these are technically what are called 'Dirichlet concentration
# paramaters', which need not take on values between 0 and 1. These values
# are added to after each trial, based on posterior beliefs about initial
# states. For example, if the agent believed at the end of trial 1 that it 
# was in the 'left better' context, then d{1} on trial 2 would be 
# d{1} = [1.5 0.5]' (although how large the increase in value is after 
# each trial depends on a learning rate). In general, higher values 
# indicate more confidence in one's beliefs about initial states, and 
# entail that beliefs will change more slowly (e.g., the shape of the 
# distribution encoded by d{1} = [25 25]' will change much more slowly 
# than the shape of the distribution encoded by d{1} = [.5 0.5]' with each 
# new observation).

# For context beliefs, we can specify that the agent starts out believing 
# that both contexts are equally likely, but with somewhat low confidence in 
# these beliefs:

d[1] = np.array([0.25, 0.25]).T # {'left better','right better'}

# For behavior beliefs, we can specify that the agent expects with 
# certainty that it will begin a trial in the 'start' state:

d[2] = np.array([1, 0, 0, 0]).T # {'start','hint','choose-left','choose-right'}

# print(f'\nPrior beliefs about initial states: {d}')



# State-outcome mappings and beliefs: A and a
# =========================================================================

#--------------------------------------------------------------------------
# Specify the probabilities of outcomes given each state in the generative 
# process (A)
# This includes one matrix per outcome modality
# Note: By default, these will also be the beliefs in the generative model
#--------------------------------------------------------------------------

# First we specify the mapping from states to observed hints (outcome
# modality 1). Here, the rows correspond to observations, the columns
# correspond to the first state factor (context), and the third dimension
# corresponds to behavior. Each column is a probability distribution
# that must sum to 1.

# We start by specifying that both contexts generate the 'No Hint'
# observation across all behavior states:

Ns = {}
Ns[1] = D[1].shape[0]  # Number of states in the first factor (2)
Ns[2] = D[2].shape[0]  # Number of states in the second factor (4)

# print(f'\nNumber of states in each factor: {Ns}')

A = {}

A[1] = np.zeros((3, Ns[1], Ns[2]))  # 3 hints, 2 contexts, 4 behaviors
A[2] = np.zeros((3, Ns[1], Ns[2]))  # 3 outcomes, 2 contexts, 4 behaviors
A[3] = np.zeros((4, Ns[1], Ns[2]))  # 4 observed actions, 2 contexts, 4 behaviors 

for i in range(Ns[2]):
    A[1][:, :, i] = np.array([[1, 1], # No hint
                              [0, 0], # Machine-left hint
                              [0, 0]])   # Machine-right hint


# Then we specify that the 'Get Hint' behavior state generates a hint that
# either the left or right slot machine is better, depending on the context
# state. In this case, the hints are accurate with a probability of pHA. 

pHA = 1.0  # Probability of hint accuracy

A[1][:, :, 1] = np.array([[0, 0], # No hint
                          [pHA, 1-pHA], # Machine-left hint
                          [1-pHA, pHA]])   # Machine-right hint

# Next we specify the mapping between states and wins/losses. The first two
# behavior states ('Start' and 'Get Hint') do not generate either win or
# loss observations in either context:

for i in range(2):
    A[2][:, :, i] = np.array([[1, 1], # Start
                              [0, 0], # Loss
                              [0, 0]])   # Win


# Choosing the left machine (behavior state 3) generates wins with
# probability pWin, which differs depending on the context state (columns):

pWin = .8  # By default we set this to .8, but try changing its value to 
           # see how it affects model behavior


A[2][:, :, 2] = np.array([[0, 0], # Start
                          [1-pWin, pWin], # Loss
                          [pWin, 1-pWin]]) # Win


# Choosing the right machine (behavior state 4) generates wins with
# probability pWin, with the reverse mapping to context states from 
# choosing the left machine:

A[2][:, :, 3] = np.array([[0, 0], # Start
                          [pWin, 1-pWin], # Loss
                          [1-pWin, pWin]]) # Win


# Finally, we specify an identity mapping between behavior states and
# observed behaviors, to ensure the agent knows that behaviors were carried
# out as planned. Here, each row corresponds to each behavior state.

for i in range(Ns[2]):
    A[3][i, :, i] = np.array([1, 1])


# print(f'\nState-outcome mappings:')
# print(A)


#--------------------------------------------------------------------------
# Specify prior beliefs about state-outcome mappings in the generative model 
# (a)
# Note: This is optional, and will simulate learning state-outcome mappings 
# if specified.
#--------------------------------------------------------------------------
           
# We will not simulate, learning the 'a' matrix here.  
# However, similar to learning priors over initial states, this simply
# requires specifying a matrix (a) with the same structure as the
# generative process (A), but with Dirichlet concentration parameters that
# can encode beliefs (and confidence in those beliefs) that need not
# match the generative process. Learning then corresponds to
# adding to the values of matrix entries, based on what outcomes were 
# observed when the agent believed it was in a particular state. For
# example, if the agent observed a win while believing it was in the 
# 'left better' context and the 'choose left machine' behavior state,
# the corresponding probability value would increase for that location in
# the state outcome-mapping (i.e., a{2}(3,1,3) might change from .8 to
# 1.8).

# One simple way to set up this matrix is by:
 
# 1. initially identifying it with the generative process 
# 2. multiplying the values by a large number to prevent learning all
#    aspects of the matrix (so the shape of the distribution changes very slowly)
# 3. adjusting the elements you want to differ from the generative process.

# For example, to simulate learning the reward probabilities, we could specify:

    # a={}
    #
    # a[1] = A[1]*200
    # a[2] = A[2]*200
    # a[3] = A[3]*200
    #
    # a[2][:, :, 2] = np.array([[0, 0]=None, # Start
    #                           [0.5, 0.5]=None, # Loss
    #                           [0.5, 0.5]]) # Win
    #
    # a[2][:, :, 3] = np.array([[0, 0]=None, # Start
    #                           [0.5, 0.5]=None, # Loss
    #                           [0.5, 0.5]]) # Win
    #
 

# As another example, to simulate learning the hint accuracy one
# might specify:

    # a={}
    #
    # a[1] = A[1]*200
    # a[2] = A[2]*200
    # a[3] = A[3]*200
    #
    # a[1][:, :, 2] = np.array([[0, 0]=None, # No hint
    #                           [0.25, 0.25]=None, # Machine-left hint
    #                           [0.25, 0.25]])   # Machine-right hint
    #




# Controlled transitions and transition beliefs : B{:,:,u} and b(:,:,u)
#==========================================================================

#--------------------------------------------------------------------------
# Next, we have to specify the probabilistic transitions between hidden states
# under each action (sometimes called 'control states'). 
# Note: By default, these will also be the transitions beliefs 
# for the generative model
#--------------------------------------------------------------------------

B={}

# Columns are states at time t. Rows are states at t+1.

# The agent cannot control the context state, so there is only 1 'action',
# indicating that contexts remain stable within a trial:

B[1] = np.zeros((Ns[1], Ns[1], 1))  # 2 states, 2 states

B[1][:, :, 0] = np.array([[1, 0], # 'left better' context -> 'left better' context
                          [0, 1]]) # 'right better' context -> 'right better' context


# The agent can control the behavior state, and we include 4 possible 
# actions:

B[2] = np.zeros((Ns[2], Ns[2], 4))  # 4 states, 4 states, 4 actions

# Move to the Start state (behavior state 1) from any other behavior state:
B[2][:, :, 0] = np.array([[1, 1, 1, 1], # any state -> 'Start' state
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]])

# Move to the 'Get Hint' state (behavior state 2) from any other behavior state:
B[2][:, :, 1] = np.array([[0, 0, 0, 0],
                          [1, 1, 1, 1], # any state -> 'Get Hint' state
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]])

# Move to the 'Choose Left' state (behavior state 3) from any other behavior state:
B[2][:, :, 2] = np.array([[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [1, 1, 1, 1], # any state -> 'Choose Left' state
                          [0, 0, 0, 0]])

# Move to the 'Choose Right' state (behavior state 4) from any other behavior state:
B[2][:, :, 3] = np.array([[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [1, 1, 1, 1]])  # any state -> 'Choose Right' state

# print(f'\nControlled transitions:')
# print(B)

#--------------------------------------------------------------------------
# Specify prior beliefs about state transitions in the generative model
# (b). This is a set of matrices with the same structure as B.
# Note: This is optional, and will simulate learning state transitions if 
# specified.
#--------------------------------------------------------------------------
          
# For this example, we will not simulate learning transition beliefs. 
# But, similar to learning d and a, this just involves accumulating
# Dirichlet concentration parameters. Here, transition beliefs are updated
# after each trial when the agent believes it was in a given state at time
# t and and another state at t+1.


# Preferred outcomes: C and c
#==========================================================================

#--------------------------------------------------------------------------
# Next, we have to specify the 'prior preferences', encoded here as log
# probabilities. 
#--------------------------------------------------------------------------

# One matrix per outcome modality. Each row is an observation, and each
# columns is a time point. Negative values indicate lower preference,
# positive values indicate a high preference. Stronger preferences promote
# risky choices and reduced information-seeking.

No={} # Number of outcomes in each modality
No[1] = A[1].shape[0]  # Number of hints (3) (no hint, left hint, right hint) 
No[2] = A[2].shape[0]  # Number of outcomes (3) (start, loss, win)
No[3] = A[3].shape[0]  # Number of actions (4) (start, hint, choose-left, choose-right)


C = {}

C[1] = np.zeros((No[1], T))  # 3 hints, 3 time points
C[2] = np.zeros((No[2], T))  # 3 outcomes, 3 time points
C[3] = np.zeros((No[3], T))  # 4 observed actions, 3 time points

# Then we can specify a 'loss aversion' magnitude (la) at time points 2 
# and 3, and a 'reward seeking' (or 'risk-seeking') magnitude (rs). Here,
# rs is divided by 2 at the third time point to encode a smaller win ($2
# instead of $4) if taking the hint before choosing a slot machine.

la = 1  # By default we set this to 1, but try changing its value to 
        # see how it affects model behavior

rs = rs1  # We set this value at the top of the script. 
          # By default we set it to 4, but try changing its value to 
          # see how it affects model behavior (higher values will promote
          # risk-seeking, as described in the main text)

C[2][:, :] = np.array([[0, 0, 0], # Start
                       [0, -la, -la], # Loss
                       [0, rs, rs/2]])  # Win

# Note that, expanded out, this means that the other C-matrices will be:

C[1] = np.array([[0, 0, 0], # No hint
                 [0, 0, 0], # Machine-left hint
                 [0, 0, 0]]) # Machine-right hint

C[3] = np.array([[0, 0, 0], # Start
                 [0, 0, 0], # Hint
                 [0, 0, 0], # Choose Left
                 [0, 0, 0]]) # Choose Right


#--------------------------------------------------------------------------
# One can also optionally choose to simulate preference learning by
# specifying a Dirichlet distribution over preferences (c). 
#--------------------------------------------------------------------------

# This will not be simulated here. However, this works by increasing the
# preference magnitude for an outcome each time that outcome is observed.
# The assumption here is that preferences naturally increase for entering
# situations that are more familiar.



# Allowable policies: U or V. 
#==========================================================================

#--------------------------------------------------------------------------
# Each policy is a sequence of actions over time that the agent can 
# consider. 
#--------------------------------------------------------------------------

# Policies can be specified as 'shallow' (looking only one step
# ahead), as specified by U. Or policies can be specified as 'deep' 
# (planning actions all the way to the end of the trial), as specified by
# V. Both U and V must be specified for each state factor as the third
# matrix dimension. This will simply be all 1s if that state is not
# controllable.

# For example, specifying U could simply be:

    # Np = 4  # Number of policies
    # Nf = 2  # Number of state factors
    #
    # U = np.ones((1, Np, Nf))
    #
    # U[:, :, 0] = np.array([0, 0, 0, 0])  # Context state (B[1]) is not controllable
    # U[:, :, 1] = np.array([0, 1, 2, 3])  # All four actions in B[2] are allowed

# For our simulations, we will specify V, where rows correspond to time 
# points and should be length T-1 (here, 2 transitions, from time point 1
# to time point 2, and time point 2 to time point 3):

Np = 5  # Number of policies
Nf = 2  # Number of state factors

V = np.ones((T-1, Np, Nf))

V[:, :, 0] = np.array([[0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]])  # Context state (B[1]) is not controllable

V[:, :, 1] = np.array([[0, 1, 1, 2, 3],
                       [0, 2, 3, 0, 0]])

# For V[:,:,1], columns left to right indicate policies allowing:
# 1. staying in the start state 
# 2. taking the hint then choosing the left machine
# 3. taking the hint then choosing the right machine
# 4. choosing the left machine right away (then returning to start state)
# 5. choosing the right machine right away (then returning to start state)


# Habits: E and e. 
#==========================================================================

#--------------------------------------------------------------------------
# Optional: a columns vector with one entry per policy, indicating the 
# prior probability of choosing that policy (i.e., independent of other 
# beliefs). 
#--------------------------------------------------------------------------

# We will not equip our agent with habits in our example simulations, 
# but this could be specified as a follows if one wanted to include a
# strong habit to choose the 4th policy:

# E = np.array([0.1, 0.1, 0.1, 0.6, 0.1]).T

# To incorporate habit learning, where policies become more likely after 
# each time they are chosen, one can also specify concentration parameters
# by specifying e. For example:

# e = np.array([1, 1, 1, 1, 1]).T





# Additional optional parameters. 
#==========================================================================

# Eta: learning rate (0-1) controlling the magnitude of concentration parameter
# updates after each trial (if learning is enabled).

eta = 0.5  # By default we here set this to 0.5, but try changing its value  
           # to see how it affects model behavior


# Omega: forgetting rate (0-1) controlling the reduction in concentration parameter
# magnitudes after each trial (if learning is enabled). This controls the
# degree to which newer experience can 'over-write' what has been learned
# from older experiences. It is adaptive in environments where the true
# parameters in the generative process (priors, likelihoods, etc.) can
# change over time. A low value for omega can be seen as a prior that the
# world is volatile and that contingencies change over time.

omega = 1  # By default we here set this to 1 (indicating no forgetting, 
           # but try changing its value to see how it affects model behavior. 
           # Values below 1 indicate greater rates of forgetting.


# Beta: Expected precision of expected free energy (G) over policies (a 
# positive value, with higher values indicating lower expected precision).
# Lower values increase the influence of habits (E) and otherwise make
# policy selection less deteriministic. For our example simulations we will
# simply set this to its default value of 1:

beta = 1  # By default this is set to 1, but try increasing its value 
          # to lower precision and see how it affects model behavior

# Alpha: An 'inverse temperature' or 'action precision' parameter that 
# controls how much randomness there is when selecting actions (e.g., how 
# often the agent might choose not to take the hint, even if the model 
# assigned the highest probability to that action. This is a positive 
# number, where higher values indicate less randomness. Here we set this to 
# a high value:

alpha = 32   # Any positive number. 1 is very low, 32 is fairly high; 
             # an extremely high value can be used to specify
             # deterministic action (e.g., 512)

# ERP: This parameter controls the degree of belief resetting at each 
# time point in a trial when simulating neural responses. A value of 1
# indicates no resetting, in which priors smoothly carry over. Higher
# values indicate degree of loss in prior confidence at each time step.

erp = 1  # By default we here set this to 1, but try increasing its value  
         # to see how it affects simulated neural (and behavioral) responses

# tau: Time constant for evidence accumulation. This parameter controls the
# magnitude of updates at each iteration of gradient descent. Larger values 
# of tau will lead to smaller updates and slower convergence time, 
# but will also promote greater stability in posterior beliefs. 

tau = 12  # Here we set this to 12 to simulate smooth physiological responses,   
          # but try adjusting its value to see how it affects simulated
          # neural (and behavioral) responses

# Note: If these values are left unspecified, they are assigned default
# values when running simulations. These default values can be found within
# the spm_MDP_VB_X script (and in the spm_MDP_VB_X_tutorial script we
# provide in this tutorial).

# Other optional constants. 
#==========================================================================

# Chi: Occam's window parameter for the update threshold in deep temporal 
# models. In hierarchical models, this parameter controls how quickly
# convergence is 'cut off' during lower-level evidence accumulation. 
# specifically, it sets an uncertainty threshold, below which no additional 
# trial epochs are simulated. By default, this is set to 1/64. Smaller 
# numbers (e.g., 1/128) indicate lower uncertainty (greater confidence) is
# required before which the number of trial epochs are shortened.

# zeta: Occam's window for policies. This parameter controls the threshold
# at which a policy ceases to be considered if its free energy
# becomes too high (i.e., when it becomes too implausible to consider
# further relative to other policies). It is set to default at a value of 
# 3. Higher values indicate a higher threshold. For example, a value of 6
# would indicate that a greater difference between a given policy and the
# best policy before that policy was 'pruned' (i.e., ceased to be
# considered). Policies will therefore be removed more quickly with smaller
# zeta values.
         
# Note: The spm_MDP_VB_X function is also equipped with broader functionality
# allowing incorporation of mixed (discrete and continuous) models,
# plotting, simulating Bayesian model reduction during simulated
# rest/sleep, among others. We do not describe these in detail here, but
# are described in the documentation at the top of the function.


# True states and outcomes: s and o. 
#==========================================================================

#--------------------------------------------------------------------------
# Optionally, one can also specify true states and outcomes for some or all
# time points with s and o. If not specified, these will be 
# generated by the generative process. 
#--------------------------------------------------------------------------

# For example, this means the true states at time point 1 are left context 
# and start state:

# s = np.array([1,
#               1]).T  # the later time points (rows for each state factor) are 0s,
#                      # indicating not specified.

# And this means the observations at time point 1 are the No Hint, Null,
# and Start behavior observations.

# o = np.array([1,
#               1,
#               1]).T  # the later time points (rows for each outcome modality) are
#                      # 0s, indicating not specified.


## 2. Define MDP structure
#==========================================================================
#==========================================================================

class MDP:
    def __init__(self,
                 T=None, # Number of time steps
                 V=None, # Allowable (deep) policies 
                 U=None, # Allowable (shallow) policies

                 A=None, # Likelihoods (state-outcome mappings)
                 B=None, # Transition probabilities (state-state mappings)
                 C=None, # Prefered outcomes (outcome preferences)
                 D=None, # Prior over initial states
                 E=None, # Habits (prior over policies)

                 # Optional concentration parameters for learning
                 a=None, # enables learning likelihoods
                 b=None, # enables learning state transitions
                 c=None, # enables learning preferences
                 d=None, # enables learning priors over initial states
                 e=None, # enables learning habits

                 # Specifying true states and outcomes
                 s=None, # True states
                 o=None, # True outcomes

                 # Specifying other optional parameters
                 eta=None, # Learning rate
                 omega=None, # Forgetting rate
                 alpha=None, # Action precision
                 beta=None, # Precision of expected free energy over policies
                 erp=None, # Degree of belief resetting at each time step
                 tau=None, # Time constant for evidence accumulation
                 chi=None, # confidence threshold for ceasing evidence accumulation
                           # in lower levels of hierarchical models
                 zeta=None, # threshold for ceasing consideration of policies

                 label=None # Optional label for the MDP
                 ):

        # Initialize variables of required parameters in the MDP structure
        self.T = T
        self.V = V
        self.U = U
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.s = s
        self.o = o
        self.eta = eta
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.erp = erp
        self.tau = tau
        self.chi = chi
        self.zeta = zeta
        self.label = label

    def __repr__(self):
        """
        Provide a simple string representation of the MDP structure
        """
        def short_repr(x):
            """
            Provide a short string representation of x
            """
            if isinstance(x, np.ndarray):
                return f'array(shape={x.shape}, dtype={x.dtype})'
            elif isinstance(x, dict):
                string = f'dict(keys={list(x.keys())})'
                keys = list(x.keys())
                for key in keys:
                    string += f'\n{key}: {short_repr(x[key])}'
                return string
            elif isinstance(x, list):
                string = f'list(len={len(x)})'
                for i, item in enumerate(x):
                    string += f'\n{i}: {short_repr(item)}'
                return string
            elif isinstance(x, (int, float, str, bool)):
                return repr(x)
            elif x is None:
                return 'None'
            else:
                return f'<{type(x).__name__}>'

        # Get the names of the variables in the MDP structure
        variables = [name for name in self.__dict__.keys() if not name.startswith('_')]
        return '\n'.join([f'{name}: {short_repr(getattr(self, name))}' for name in variables])


# MDP_check that verifies the consistency of the main components of an
# MDP structure with respect to matrix dimensions.
# You may wish to expand or refine the checks, depending on
# exactly how strict or how broad you want them to be
# (e.g., whether to check if probabilities sum to 1, etc.).
def MDP_check(mdp):
    """
    Check the dimensional consistency of the main MDP parameters:
      - D and d (initial state priors)
      - A and a (likelihood matrices)
      - B and b (transition matrices)
      - C and c (preferences)
      - V (deep policies) or U (shallow policies)
    This function raises a ValueError if a mismatch is found,
    otherwise it prints a success message.
    """

    # ----------------------------------------------------
    # 1. Check D (initial state priors) dimension
    # ----------------------------------------------------
    if not isinstance(mdp.D, dict) or len(mdp.D) < 1:
        raise ValueError('mdp.D must be a dictionary with at least one factor')


    # For each state factor i in D, store the number of states Ns[i].
    # We assume the dict keys in mdp.D are consecutive integers starting from 1.
    Ns = {}
    for factor_i in mdp.D.keys():
        D_factor = mdp.D[factor_i]
        # Check that D_factor is a 1D numpy array
        if not (isinstance(D_factor, np.ndarray) and D_factor.ndim == 1):
            raise ValueError(f'mdp.D[{factor_i}] must be a 1D numpy array')
        Ns[factor_i] = D_factor.shape[0]


    # ----------------------------------------------------
    # 2. Check A (likelihood matrices) dimension
    # ----------------------------------------------------
    # mdp.A is a dict with one entry per outcome modality.
    # For each modality m, A[m] must be an array of shape
    # (No[m], Ns[1], Ns[2], ..., Ns[Nf]),
    # i.e. # of outcomes x [# of states in each factor].
    if not isinstance(mdp.A, dict) or len(mdp.A) < 1:
        raise ValueError('mdp.A must be a dictionary with at least one outcome modality')
    for modality_m in mdp.A.keys():
        A_m = mdp.A[modality_m]
        # For multiple discrete state factors, A can be an N-dimensional array:
        #   dimension 0: # of outcomes in this modality
        #   dimension 1..Nf: # of states in each state factor
        # e.g., for Nf=2, shape should be (# outcomes, Ns[1], Ns[2])
        if not isinstance(A_m, np.ndarray) or A_m.ndim < 2:
            raise ValueError(f'mdp.A[{modality_m}] must be a numpy array with at least 2 dimensions')

        # Required rank = 1 (outcomes) + # of state factors
        required_rank = 1 + len(Ns)
        if A_m.ndim != required_rank:
            raise ValueError(
                f'mdp.A[{modality_m}] must be a {required_rank}-dimensional array, but has {A_m.ndim} dimensions'
                )

        # Check that shape matches # states per factor
        # and also check the number of outcomes in first dimension.
        # We'll call the first dimension # outcomes, so skip that here:
        for i, factor_i in enumerate(sorted(Ns.keys()), start=1):
            if A_m.shape[i] != Ns[factor_i]:
                raise ValueError(
                        f'mdp.A[{modality_m}] shape mismatch at dimension {i}:'
                        f'expected {Ns[factor_i]} states in factor {factor_i},'
                        f'found {A_m.shape[i]} in shape {A_m.shape}.'
                        )
                    


    # ----------------------------------------------------
    # 3. Check B (transition matrices) dimension
    # ----------------------------------------------------
    # mdp.B is a dict with one entry per state factor (same keys as D).
    # B[i] has shape (Ns[i], Ns[i], Nu[i]) where Nu[i] = number of control states for that factor.
    # (If the factor is uncontrollable, we often have B[i].shape == (Ns[i], Ns[i], 1))
    if not isinstance(mdp.B, dict) or len(mdp.B) < 1:
        raise ValueError('mdp.B must be a dictionary with at least one factor')

    for factor_i in mdp.B.keys():
        B_factor = mdp.B[factor_i]
        if not isinstance(B_factor, np.ndarray):
            raise ValueError(f'mdp.B[{factor_i}] must be a numpy array')
        if B_factor.ndim != 3:
            raise ValueError(
                f'mdp.B[{factor_i}] dimension mismatch: expected shape '
                f'({Ns[factor_i]}, {Ns[factor_i]}, #actions), '
                f'got shape {B_factor.shape}'
                )
        # We do not strictly check the third dimension here (the # of actions),
        # because it can vary from 1 up to however many control states you define.

    # ----------------------------------------------------
    # 4. Check C (preferences) dimension
    # ----------------------------------------------------
    # mdp.C is a dict with one entry per outcome modality.
    # Typically for a discrete-time model with T time steps,
    # C[m] has shape (# outcomes in modality m, T) or possibly (# outcomes in modality m, 1).
    if mdp.C is not None:
        if not isinstance(mdp.C, dict):
            raise ValueError('mdp.C must be a dictionary if provided')
        for modality_m in mdp.C.keys():
            C_m = mdp.C[modality_m]
            if not isinstance(C_m, np.ndarray):
                raise ValueError(f'mdp.C[{modality_m}] must be a numpy array')
            # Check that the 1st dimension is # outcomes in this modality (we can compare with A)
            # from the shape of mdp.A[modality_m]. The 1st dimension in A is # outcomes.
            A_m = mdp.A[modality_m]
            num_outcomes = A_m.shape[0]
            if C_m.shape[0] != num_outcomes:
                raise ValueError(
                    f'mdp.C[{modality_m}] shape mismatch: expected {num_outcomes} outcomes, (rows of A), '
                    f'(# outcomes in A), got {C_m.shape[0]} outcomes in shape {C_m.shape}'
                    )
            # The second dimension typically matches mdp.T, but might also be 1
            # or sometimes T if the user wants time-varying preferences.
            if (C_m.shape[1] != mdp.T) and (C_m.shape[1] != 1):
                raise ValueError(
                    f'mdp.C[{modality_m}] shape mismatch: the second dimension should be either 1 or T={mdp.T}, '
                    f'but got {C_m.shape[1]} in shape {C_m.shape}'
                    )

    # ----------------------------------------------------
    # 5. Check V (deep policies) or U (shallow policies)
    # ----------------------------------------------------
    # If V is not None, it should have shape (T-1, #policies, #factors).
    # If U is not None, it should have shape (1, #policies, #factors) for shallow planning.
    # We'll do a basic check if either is present.

    # Check V first
    if mdp.V is not None:
        if not isinstance(mdp.V, np.ndarray):
            raise ValueError('mdp.V must be a numpy array if provided')
        if mdp.V.ndim != 3:
            raise ValueError(
                f'mdp.V dimension mismatch: expected shape (T-1, #policies, #factors), '
                f'got shape {mdp.V.shape}'
                )
        if mdp.V.shape[0] != mdp.T - 1:
            raise ValueError(
                f'mdp.V shape mismatch: expected {mdp.T - 1} time steps, got {mdp.V.shape[0]}'
                )
        # The second dimension is #policies, which can vary, so no strict check here
        # The third dimension is #factors, which should be len(Ns)
        if mdp.V.shape[2] != len(Ns):
            raise ValueError(
                f'mdp.V shape mismatch: expected {len(Ns)} state factors (third dimension), '
                f'got {mdp.V.shape[2]}'
                )

    # Check U next
    if mdp.U is not None:
        if not isinstance(mdp.U, np.ndarray):
            raise ValueError('mdp.U must be a numpy array if provided')
        if mdp.U.ndim != 3:
            raise ValueError(
                f'mdp.U dimension mismatch: expected shape (1, #policies, #factors), '
                f'got shape {mdp.U.shape}'
                )
        if mdp.U.shape[0] != 1:
            raise ValueError(
                f'mdp.U shape mismatch: expected 1 time step, got {mdp.U.shape[0]}'
                )
        # The second dimension is #policies, which can vary, so no strict check here
        # The third dimension is #factors, which should be len(Ns)
        if mdp.U.shape[2] != len(Ns):
            raise ValueError(
                f'mdp.U shape mismatch: expected {len(Ns)} state factors (third dimension), '
                f'got {mdp.U.shape[2]}'
                )

    # ----------------------------------------------------
    # If we reach this point, all checks have passed
    # ----------------------------------------------------
    print('MDP structure check passed successfully!')
    print('MDP structure is consistent with respect to matrix dimensions.')





# Create MDP structure
label = {}
label['factor'] = {}
label['factor'][1] = 'contexts'
label['factor'][2] = 'choice states'
label['name'] = {}
label['name'][1] = ['left better', 'right better']
label['name'][2] = ['start', 'hint', 'choose-left', 'choose-right']
label['modality'] = {}
label['modality'][1] = 'hint'
label['modality'][2] = 'outcome'
label['modality'][3] = 'observed action'
label['outcome'] = {}
label['outcome'][1] = ['no hint', 'left hint', 'right hint']
label['outcome'][2] = ['start', 'loss', 'win']
label['outcome'][3] = ['start', 'hint', 'choose-left', 'choose-right']
label['action'] = {}
label['action'][2] = ['start', 'hint', 'left', 'right']





mdp = MDP(T=T,
          V=V,
          A=A,
          B=B,
          C=C,
          D=D,
          E=None,
          a=None,
          b=None,
          c=None,
          d=None,
          e=None,
          s=None,
          o=None,
          eta=eta,
          omega=omega,
          alpha=alpha,
          beta=beta,
          erp=erp,
          tau=tau,
          chi=None,
          zeta=None,
          label=label)

print(mdp)

MDP_check(mdp)





