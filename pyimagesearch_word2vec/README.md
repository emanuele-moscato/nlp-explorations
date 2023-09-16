# Word2Vec and word embeddings

Source: [here](https://pyimagesearch.com/2022/07/11/word2vec-a-study-of-embeddings-in-nlp/)

Entrypoints: scripts
- `train_CBOW.py`: obtain word embeddings using Word2Vec via the continuous bag-of-words (CBOW) algorithm.
- `train_SkipGram.py`: obtain word embeddings using Word2Vec via the Skip-gram algorithm.

Both scripts produce plots (saved in the `outputs/` directory) for the loss function as a function of the epoch number and for the embeddings of some of the words reduced to 2 dimensions using TSNE.
