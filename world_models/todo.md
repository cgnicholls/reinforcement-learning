# To do list
Items marked with '-' are pending. Items striked through are complete.

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
