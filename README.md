# Customized Attention Span (CAS) Models

The recurrent weighted average (RWA) model is a new kind of recurrent neural network (RNN) that is based on the attention mechanism. The model is described  in this [manuscript](https://arxiv.org/abs/1703.01253). The RWA model exhibits no bias as to where information is located along a sequence. Recent information is just as important as information at the beginning of the sequence. For many problems, this is undesirable. That is why the RWA model needs an *attention span*.

In this repository, the RWA model is reimplemented as a TensorFlow RNNCell. The implementation of the RWA model can be used as easily as any other RNN architecture available in TensorFlow. To enforce an attention span, a decay term has been introduced. All decay values must be between 0 and 1. When the decay value is 1, all information along a sequence is treated with equal importance. When the decay value is close to 0, only the most recent information is used.

Each unit can be given a different decay value. Some units can have very short attention spans while other units can have indefinite attention spans. It is my hope that this model will exhibit superior performance on a range of problems in fields like NLP and bioinformatics.

