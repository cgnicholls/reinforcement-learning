# To do list
Items marked with '-' are pending. Items marked with 'x' are complete.

x Can collect rollouts from a game to train the VAE.
- Can load in rollouts. (This is basically done but needs more testing).
- Collect 10,000 rollouts from Pong
- Train VAE on the rollouts.
- Should be able to save and load the VAE.
    - Ideally can do this within a bigger graph, so that we can later have the VAE as part of a big graph 
    where we also have the MDN-RNN and controller.
    - The test is just: can initialise VAE, can save VAE, can restore VAE.