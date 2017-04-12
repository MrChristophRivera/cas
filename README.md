# Decaying Recurrent Weighted Average (DRWA)

I recently published a new kind of recurrent neural network (RNN) model called a recurrent weighted average (RWA). The method is described here: https://arxiv.org/abs/1703.01253.

One disadvantage of the RWA model is that is shows no bias toward recent information. All information along a sequence is treated with equal importance. To correct this, a decay term has been introduced. All decay values must be between 0 and 1. With the decay value is 1, all information along a sequence is treated with equal importance. When the decay value is close to 0, only the most recent information is used.

Each unit can be given a different decay value. That way, some units can be focused on just new information, while other units can use information present in the deep past.

