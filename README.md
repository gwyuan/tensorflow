# Tensorflow

http://web.stanford.edu/class/cs20si/


Tensorflow is a machine learning system that operates at large scale and in heterogeneous environments.
TF is not the only deep learning library.

__Getting start with__

```
import tensorflow as tf
```

__tf.Session()__
A Session object encapsulates the environment in which Operation objects are executed, and Tensor objects are evaluated.


# NLP:
http://web.stanford.edu/class/cs224n/

## Word2Vec
https://github.com/tensorflow/models/tree/master/tutorials/embedding

(Mikolov, et. al.) [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/abs/1301.3781),
ICLR 2013.

Detailed instructions on how to get started and use them are available in the
tutorials. Brief instructions are below.

* [Word2Vec Tutorial](http://tensorflow.org/tutorials/word2vec)

Assuming you have cloned the git repository, navigate into this directory. To download the example text and evaluation data:

```shell
curl http://mattmahoney.net/dc/text8.zip > text8.zip
unzip text8.zip
curl https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip > source-archive.zip
unzip -p source-archive.zip  word2vec/trunk/questions-words.txt > questions-words.txt
rm text8.zip source-archive.zip
```

You will need to compile the ops as follows:

```shell
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
g++ -std=c++11 -shared word2vec_ops.cc word2vec_kernels.cc -o word2vec_ops.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0
```

On Mac, add `-undefined dynamic_lookup` to the g++ command.

(For an explanation of what this is doing, see the tutorial on [Adding a New Op to TensorFlow](https://www.tensorflow.org/how_tos/adding_an_op/#building_the_op_library). The flag `-D_GLIBCXX_USE_CXX11_ABI=0` is included to support newer versions of gcc. However, if you compiled TensorFlow from source using gcc 5 or later, you may need to exclude the flag.)
Then run using:

```shell
python word2vec_optimized.py \
  --train_data=text8 \
  --eval_data=questions-words.txt \
  --save_path=/tmp/
```
