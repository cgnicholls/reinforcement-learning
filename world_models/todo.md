# To do list
Items marked with '-' are pending. Items striked through are complete.

- We can test how well each part does individually
    - Replace VAE with just some well-known good features. E.g. for Pong - just
     the (x, y) location of the ball and the y positions of the bats (note the
     MDN-RNN should learn velocities).
    - Replace VAE and MDN-RNN with good features - include velocity with the
    above features.

- Can train the MDN-RNN.
    - Write the MDN-RNN.
- Save and load VAE in main
- Tensorboard logging
- Can we load it all into memory?
- Ideally can do this within a bigger graph, so that we can later have the VAE as part of a big graph where we also have the MDN-RNN and controller.

Less important:
- Investigate why we get the weird error when loading using deepdish sometimes.
- Make all minibatches the same size.

Done:
- ~~Can collect rollouts from a game to train the VAE.~~
- ~~Can load in rollouts.~~
- ~~Collect 10,000 rollouts from Pong.~~
    - ~~Can continue loading rollouts from an existing directory.~~
- ~~Train VAE on the rollouts.~~
