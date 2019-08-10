# trader

uses a policy gradient method with a CNN to learn how to manage a portfolio of crypto currencies.

It is based on a paper "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem" by Zhengyao Jiang,
Dixing Xu, Jinjun Liang https://arxiv.org/pdf/1706.10059.pdf

Some differences are a larger network, a different activation function and way of normalizing the quotes.
The model is a 5 layered CNN with tanh activation (since relu doesn't work).
Each crypto's quote is also run to its own lstm, this increases training time. But seems to make it behave less like a stock jockey.

It works well on reasonably complex syntetic data that has some randomness in its price. This includes a transaction fee of 0.2%.
Tried batch normalization on several occasions but this prevents the model from converging. So its not used.
I have not yet trained it for a longer period of time on real data, with which it still strugles.
So the stellar experimental results from the paper still elude me :D


#To download data from poloniex use the download function in get_data.py

#to run: python3 crypto_ai.py

Dependencies:
 - tensorflow
 - numpy
 - matplotlib
 - pickle
