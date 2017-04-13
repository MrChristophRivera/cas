# Customized Attention Span (CAS) Models

The recurrent weighted average (RWA) model is a new kind of recurrent neural network (RNN) that is based on the attention mechanism. The model is described in this [manuscript](https://arxiv.org/abs/1703.01253). The RWA model exhibits no bias as to where information is located along a sequence. Recent information is just as important as information at the beginning of the sequence. For many problems, this is undesirable. That is why the RWA model needs an *attention span*.

In this repository, the RWA model is reimplemented as a TensorFlow RNNCell. The implementation of the RWA model can be used as easily as any other RNN architecture available in TensorFlow. To enforce an attention span, a decay term has been introduced. The value of the decay term determines how quickly a memory is forgotten and is inversely proportional to the expected half-life. When the decay term is 0 memories are retained indefinitely, although more important information can still overwrite less important information. When the decay term is larger than 0, the memory will be forgotten. The larger the decay term, the quicker the model forgets.

Each unit can be given a different decay value. Some units can have very short attention spans while other units can have indefinite attention spans. It is my hope that this model will exhibit superior performance on a range of problems in fields like NLP and bioinformatics.

