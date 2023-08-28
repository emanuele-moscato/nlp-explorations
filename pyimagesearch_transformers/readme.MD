# Transformers with Tensorflow and Keras

This project is taken from the 3-part blog post series released by [pyimagesearch.com](https://www.pyimagesearch.com) with title **A deep dive into Transformers with Tensorflow and Keras**. Links:
- [Part 1](https://pyimagesearch.com/2022/09/05/a-deep-dive-into-transformers-with-tensorflow-and-keras-part-1/)
- [Part 2](https://pyimagesearch.com/2022/09/26/a-deep-dive-into-transformers-with-tensorflow-and-keras-part-2/)
- [Part 3](https://pyimagesearch.com/2022/11/07/a-deep-dive-into-transformers-with-tensorflow-and-keras-part-3/)

In turns, the third blog post (focused on the Tensorflow implementation of the model) had [this](https://www.tensorflow.org/text/tutorials/transformer) official Tensorflow tutorial as its main source for code, with some modifications.

Because the code from pyimagesearch was not working (maybe due to a different version of Tensorflow, I didn't check), I had to adapt it a bit, in particular:
- At inference time (i.e. in the `Translator` object) there didn't seem to be any reference to the softmax function that should map the logit outputted by the trasformer in probabilities over the target vocabulary (this wasn't present in the Tensorflow tutorial either). Indeed, the final dense layer in the `Transformer` didn't have any activation function, and its output is passed to the argmax function right after it's computed within the `Translator` - either I'm missing something here or the armax is in fact not acting on probabilities. I added a softmax before the argmax and things seem to work.
- There was a shape mismatch in `Translator.call` when the new `PredictedId` is added to the `outputArray`: originally its 0-th entry (it's a 1-dimensional tensor with 1 component) was selected, but this gave me error. If we don't select any component (i.e. we just append `predictedId` as, for example, `[22]`) it just works.

**Note:** the absence of softmax is a problem (provided I'm getting this right) only at inference time because at training time we use a `SparseCategoricalCrossentropy` loss for which we specify `from_logits=True`, which allows us to pass it the output of the `Transformer` before applying the softmax function.
