# coding: utf-8
# Implements the asynchronous Q-learning algorithm.

# Need shared variables between threads

# Pseudocode (for each actor-learner thread):
# Assume global shared theta, theta- and counter T = 0.
# Initialize thread step counter t <- 0
# Initialize target network weights theta- <- theta
# Initialize network gradients dtheta <- 0
# Get initial state s
# repeat
#   take action a with epsilon-greedy policy based on Q(s,a;theta)
#   receive new state s' and reward r
#   y = r (for terminal s')
#     = r + gamma * max_a' Q(s', a', theta-) (for non-terminal s')
#   accumulate gradients wrt theta: dtheta <- dtheta + grad_theta(y-Q(s,a;theta))^2
#   s = s'
#   T <- T + 1 and t <- t + 1
#   if T mod I_target == 0:
#       update target network theta- <- theta
#   if t mod I_asyncupdate == 0 or s is terminal:
#       perform asynchronous update of theta using dtheta
#       clear gradients dtheta <- 0
# until T > T_max

# First step is to implement one thread and get it running on its own.
def thread():
    t = 0
    thetam = theta
    dtheta = 0
    
    
