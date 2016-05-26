This directory will contain files with stuff learned during the training.
Specifically:
* `network.th7`: The model with learned parameters.
* `stats.th7`: Various data used during the training, e.g. action counter or points of the plots.
* `memory.sqlite`: Replay memory database.

The replay memory can be reused for different training sessions. `network.th7` and `stats.th7` have to be deleted if you want to train a new model.
