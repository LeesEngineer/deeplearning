<img width="2304" height="1080" alt="QQ_1778326775514" src="https://github.com/user-attachments/assets/89dcb5de-5329-472d-bd8f-411efe570446" />

<p>All of these problems can be addressed as supervised learning with label data (X, Y) as the training set. In some situation, both the X and Y are sequences.</p>

</br>

# Notation

</br>

<p>If you want a sequence model to automatically tell you where are the people's names in the sentence. The problem is called named-entity recognition.</p>

<p>X: Harry Potter and Hermione Granger invented a new spell. Y: 110110000. We use the index x<t> and y<t>.</p>

<p>To represent a word in the sentence, the first you do is coming up with a <b>vocabulary</b>, that means making a list of the words that you will use</p>

<img width="456" height="686" alt="QQ_1778340659633" src="https://github.com/user-attachments/assets/f8986614-fe92-419a-8578-484fcd56cc4a" />

<p>A lot of commercial app use dictionary sizes of 30,000 or 50,000 words.</p>

<p>One way to build this dictionary is looking through your training sets and find the top 10,000 occurring words. Then use one-hot representation to represent each of these words. Every vector is a 10,000 dimensional vector.</p>

<p>If you encounter a word that is not in your vocabulary, then create a new token or a new fake word called unknown word which notes as <UNK>.</p>

</br>

# Recurrent Neural Network Model

</br>

<p>Let's talk about how you can build a model to learn the mapping from X to Y.</p>

<p>One thing you can do is build a standard nn.</p>

<img width="1640" height="592" alt="QQ_1778401424879" src="https://github.com/user-attachments/assets/f8ee235d-b922-4a0b-be77-2f77bf2e913b" />

<p>You could feed nine one-hot vectors into the nn. But this turns out not to work well. There are two main problems.</p>

- The inputs and outputs can be different lengths in different examples.

- Doesn't share features learned across different positions of text

<p>In particular, if the nn has learned the word like Harry appearing in position one. But if Harry appears in other positions, nn needs to relearn it.</p>

<p>Let's build one RNN.</p>


<p>Feed x<1> into a nn, and get the prediction. When you goes on to read, instead of just predicting y<2> using only x<2>, it also input some information from what had computed that time-step one's. Using vector of zero as a fake time zero activation.</p>

<img width="2294" height="778" alt="QQ_1778426870266" src="https://github.com/user-attachments/assets/74016463-6bf3-4d27-87bc-044c77e3d32c" />

<p>RNN scans through the data from left to right, and the parameters it uses for each time step are shared.</p>

<p>One limitation of this nn is that the prediction at a certain time uses information from the inputs earlier in the sequence.</p>

</br>

## Forward Propagation

</br>

```
a<0> = vector of all zeros
a<1> = g(w_aa a<0> + w_ax x<1> + ba)
yhat<1> = g(w_ya a<1> + by)
```

<p>The activation function used in to compute activation will often be a <b>tanh</b>. The activation function used in to compute y depends on what your output y is. If it's a bianry classificaation problem, sigmoid is a nice choice, or could be a softmax if you have a k-way classification.</p>

<img width="1442" height="526" alt="QQ_1778497193482" src="https://github.com/user-attachments/assets/e7653f69-f2c0-428a-a362-fa95178dd824" />

<p>If a<t> is 100 dimensionals and x<t> is 10,000, W_aa is (100, 100) and W_ax is (100, 10,000). Wa would be (100, 10100). </p>

<img width="1630" height="410" alt="QQ_1778497440285" src="https://github.com/user-attachments/assets/b551d040-8858-4255-a763-52aabe19a04b" />

</br>

## Backpropagation through timw

</br>

<img width="2576" height="1012" alt="QQ_1778498322340" src="https://github.com/user-attachments/assets/50ea40aa-3993-464a-9b6d-e14fc6585a8d" />

`L<t>(yhat<t>, y<t>) = -y<t> log yhat<t> - (1 - y<t>) log (1 - yhat<t>)`

<p>L is equal to the sum of L<t>.</p>

<p>You compute the gradient through the activation chain.</p>

</br>

# DIfferent Types of RNN

</br>

<p>You've seen an RNN architecture where Tx is equal to Ty.</p>

<p>When you write a movie comment:</p>

<img width="1140" height="994" alt="QQ_1778499334245" src="https://github.com/user-attachments/assets/99c6280a-2e33-403c-9bd9-999094b88cba" /> 

<p>Rather than having to use an output at every single time-step, you can have the RNN output y at the last time-step. This is many-to-one architecture.</p>

<hr>

<p>You can also have a one-to-many architecture.</p>

<img width="1110" height="694" alt="QQ_1778499790032" src="https://github.com/user-attachments/assets/8732f29b-54f6-4db5-9554-5248fb99c3e4" />

<p>When you actually generate sequences, you take the previously synthesized output and feed it to the next layer.</p>

<hr>

<p>When Tx and Ty are at the different length, you need an alternative many-to-many architecture.</p>

<img width="1710" height="792" alt="QQ_1778500230465" src="https://github.com/user-attachments/assets/ba86ddc6-24d4-4b17-b146-79b9456d2082" />

<p>First, read the inputs. And having done that, you then have the neural network output the translation. The first part is an encoder, and the second part is a decoder.</p>

<img width="2076" height="1004" alt="QQ_1778501023820" src="https://github.com/user-attachments/assets/fec03693-ee55-4cf5-bdb9-6af8fa0da65b" />

</br>

# Language Model and Sequence Generation

</br>

<p>Use RNN to build a language model.</p>

<p>Speech Recognition:</p>

- The apple and pair salad

- The apple and pear salad

<p>When you say a sentence, the second is much more likely. A good speech reconition would output rightly even though the two sentences sound exactly the same. The way to let the system pick the second sentence is using a language model, which tells it what the probablity of either of two sentences are.</p>

```
P(first) = 3.2 * 10^-13
P(second) = 5.7 * 10^-10
```

<p>Language model output a sequence, and we call it `y<1> y<2> y<3> ... y<Ty>`</p>

<p>What a language model does is estimating the probablity of that particular sequence</p>

`P(y<1>, y<2>, y<3>, ..., y<Ty>)`

<hr>

<p>Training Set: large corpus of English texts.</p>

<p>Let's say you get a sentence as followed: "Cats average 15 hours of sleep a day". The first you do is to <b>tokenize the sentence (form a vocabulary: map these words to one-hot vectors)</b>. You also need to add an extra token called <EOS> to indicate the end.</p>

<p>When you do tokenization step, you can decide whether or not the period should be a token as well.</p>

<p>If there is a word that isn't in your vocabulary, you can <b>replace</b> it with a unique token <UNK>. Then build a RNN to model the chance of these different sequences.</p>

<p>At time zero, you're going to compute a<1>. x<1> and a<0> is all zero. <b>What a<1> does is making a softmax prediction to try to figure out what is the probablity of the first word yhat<1></b>.</p>

<p>Then the RNN steps forward to the next step, trys to figure out what is the second word. But we will <b>give it the correct first word (Cats), so that's yhat<1>. This is why y<1> is equal to x<2>. The output is again predicted by a softmax.</b></p>

<p><b>RNN only given that what had came previously. RNN outputs P(____ | "Cats"), P(____ | "Cats average") (conditional probablity)</b>. Such as, given the first three words, what is the distribution over the next word.</p>

`P(y<1>, y<2>, y<3>) = P(y<1>) * P(y<2> | y<1>) * P(y<3> | y<1>, y<2>)`

<p>P(The apple and pear salad) > P(The apple and pair salad)</p>

<p>At the end, this happens to be the EOS token. There is a high chance of P(<UNK> | "<The sentence>")</p>

<img width="2286" height="696" alt="QQ_1778583983461" src="https://github.com/user-attachments/assets/b5c905b2-102f-4cfb-b171-fe0af9885700" />

<p>Loss function: </p>

```
L(yhat<t>, y<t>) = - \sum_i y_i^<t> log yhat_i^<t>
L = \sum_t L<t>(yhat<t>, y<t>)
```

</br>

# Sampling Novel Sequence

</br>

<p>After you train a sequence model, one of the ways you can informally get a sense of what is learned is to have a sampling novel sequence.</p>

<img width="1648" height="514" alt="QQ_1778591227552" src="https://github.com/user-attachments/assets/34faf071-95cd-4dc4-989d-0ea8f79614d6" />

<p>For sampling, you do something different. First, sample what is the first word you want your model to generate. x<1> and a<0> are equal to zero. You get softmax probablity over possible outputs. Then you randomly sample according to this softmax distribution. You get P(a) or P(aaron) or P(zulu).</p>

<p>Then take the yhat<1> you just sampled and pass that to x<2> to get yhat<2>. Keep sample to get P(____ | ...)</p>

<p>Repeat until an <EOS> is encountered or the set number of timesteps is reached. You can reject any sample that came out <UNK> token and resample from the rest, or you can just leave it in the output.</p>

<hr>

<p>You can also use character level language model ([a, b, ..., 9]). But the main disadvantage of it is that you end up with much longer sequences. many sentences will have 10 to 20 words but may have many dozens of characters. <b>So character level language model is not as good as word level language model at capturing long range dependencies between how the earlier parts of the sentence affect the later parts of the sentence.</b></p>

<p>And it's more computationally expensive to train. But in some special cases we look at more character level models, <b>where you might need to deel with unknown words a lot or you have a more specialized vocabulary.</b></p>

<img width="1130" height="662" alt="QQ_1778594117361" src="https://github.com/user-attachments/assets/42d599da-fabf-43ef-a4b4-d42c4767c565" />

</br>

# Vanishing gradients  with RNNs

</br>

<img width="1674" height="574" alt="QQ_1778639879119" src="https://github.com/user-attachments/assets/df0be216-1e4a-44e4-89e1-057248864911" />

<p>You have these two sentence: "The cat, which already ate..., was full." and "The cats, which ..., were full."</p>

<p>Cat--was Cats--were</p>

<p>This is one example of when language can have very long-term dependencies. Where its word at this much earlier can affect what needs to come much later in the sentence. The RNN we have seen so far is not very good at <p>capturing very long-term dependencies.</p></p>

<img width="2158" height="324" alt="QQ_1778641256556" src="https://github.com/user-attachments/assets/4295f06b-1b12-4fc6-b876-00aa6baa2313" />

<p>To explain why, you might remember from ourearly discussion of training very deep neural network, that we talked about the vanishing gradients. When you run forward propagation and backword propagation through this very deep nn, <b>the gradient from output y would have a very hard time propagating back to affect the weights of these earlier layers to affect the computations in the earlier layers.</b></p>

<p>For an RNN, because of the same vanishing gradients problem, it might be difficult to get a nn to realize that it needs to memorize a single noun or a plural noun, so that later on in the sequence that can generate either was or were. The stuff in the middle could be arbitrarily long.</p>

<p>Because of this problem the basic RNN model has many <b>local influences</b>, meaning that the output <b>yhat3 is mainly influenced by values close to yhat3</b>.</p>

<p><b>It's difficult for the output to be strongly influenced by an output that was very early in the sequence.</b></p>

<p><b>Because whatever the output is, it's very difficult for the area to back propagate to the beginning of the sequence.</b></p>

<p>When exploding gradients happens, it can be catastrophic because the exponentially large gradients can cause your parameters to become so large that your nn parameters get really  messed up.</p>

<p>If you do find exploding gradients, one solution is applying <b>gradient clipping</b>. Look at your gradient vectors, if a number is bigger than some threshold, then <b>re-scale your gradient vector (clips according to some maximum values)</b>. This is a relatively robust solution.</p>

</br>

# Gated Recurrent Unit (GRU)

</br>

<p>GRU is a modification to the RNN hidden layer which makes it much better capturing long range connections and helps a lot with the vanishing gradient problem.</p>

```
a<t> = g(W_a[a<t-1>, x<t>] + b_a)
```

<img width="1000" height="966" alt="QQ_1778852498633" src="https://github.com/user-attachments/assets/0a2882c6-63d5-4165-a190-0b8b6fa0ddb0" />

<p>As we read the sentence "The cat, which already ate ..., was full" from left to right, the GRU unit is going to have a new variable called c which stands for memory cell. It provides a bit of memory to remember, for example, whether cat was singular or plural.</p>

<p>At time t, we have the memory cell c<t> = a<t>. At every time-step, we're going to consider overwriting the memory cell with a value ctilde<t>.</p>

`ctilde<t> = tanh(W_c [c<t-1>, x<t>] + b_c)`

<p>The idea of GRU is that we have a gate called gamma_u (u stands for update gate). The value is zero or one. The following equation is the main part of GRU.</p>

`gamma_u = sigmoid(W_u [c<t-1>, x<t>] + b_u)`

<p>For most of the possible ranges of the input, the sigmoid function is either very close to 1 or very close to 0. </p>

<p>We use gamma_u to denote the gate. We are thinking of updating c using ctilde, then <b>the gate will decide whether or not we actually update it.</b></p>

<p>Maybe this memory cell c is going to be set to either zero or one whether the subject is singular or plural. Then GRU would memorize the value of the c<t> until it was faced with the choice between was or were. If c<t> is equal to one, then use the choice was. </p>

<p>The job of gamma_u is decide when do you update these values. For example, when you see the phrase "the cat", that would be a good time to update this bit. Then when you're done using it, I don't need to memorize anymore.</p>

`c<t> = gamma_u ctilde<t> + (1 - gamma_u) c<t-1>`

<p>If gate is one, then go ahead and update that bit.</p>

<img width="1058" height="282" alt="QQ_1778930948549" src="https://github.com/user-attachments/assets/a7e04b25-269f-405f-b09b-c9fed546fd85" />

- The larger the Gamma_u: The more readily "new content" is trusted

- The smaller the Gamma_u: The more "previous memories" are retained

<img width="1126" height="768" alt="QQ_1778931682116" src="https://github.com/user-attachments/assets/83a1825b-fd90-4618-b3b5-4d7aaad67304" />

<p>Gate is quite easy to set to zero. So c<t> is very very close to c<t-1>, which is helpful for maintaining the value of cell and <b>makes it doesn't suffer from vanishing gradient problem.</b>.</p>

<p>This allows a nn runs on long range dependencies.</p>

<p>c<t>, ctilde<t> and gamma_u have the same dimension. The value in gamma_u tells you which are the bits you want to update, so that <b>you can keep some bits constant while update other bits</b>. </p>

<p>Some bits tells you the <b>singular or plural cats</b>, and <b>some bits are used to realize that you're talking about food.</b></p>

</br>

## Full GRU

</br>

```
ctilde<t> = tanh(W_c [gamma_r * c<t-1>, x<t>] + b_c)

gamma_u = sigmoid(W_u [c<t-1>, x<t>] + b_u)

gamma_r = sigmoid(W_r [c<t-1>, x<t>] + b_r)

c<t> = gamma_u ctilde<t> + (1 - gamma_u) c<t-1>
```

<p>We add one more gate gamma_r, and you can think of r as standing for relevance. We call gamma_r as <b>reset gate</b>.The gate tells you how relevant is c<t-1> to c<t>.</p>

<p><b>The purpose of gamma_r is to determine how many old memories to refer to when generating new memories.</b></p>

<hr>

<p>Gamma_u does indeed frequently approach 0 or 1. But it's a vector, so different dimensions in c<t> handle different information. Therefore:</p>

- Some dimensions are not updated for a long time.

- Some dimensions are updated every round.

- Some dimensions are updated slowly.

</br>

# LSTM (long short term memroy) Unit

</br>

<p>LSTM is more powerful than the GRU.</p>

<p>In LSTM, we will no long have the case that a<t> is equal to c<t>. And we're not using gamma_r.</p>

<p>One new property of LSTM is that instead of having one update gate, we're going to replace "gamma_u" and "1 - gamma_u" with two separate gate. We call the another gate as forget gate.</p>

<p>And we have a new output gate.</p>

```
ctilde<t> = tanh(W_c [gamma_r * a<t-1>, x<t>] + b_c)

gamma_u = sigmoid(W_u [a<t-1>, x<t>] + b_u)

gamma_f = sigmoid(W_f [a<t-1>, x<t>] + b_f)

gamma_o = sigmoid(W_o [a<t-1>, x<t>] + b_o)

c<t> = gamma_u * ctilde<t> + gamma_f * c<t-1>

a<t> = gamma_o * c<t>
```





























































