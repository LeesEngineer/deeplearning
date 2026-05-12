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

# 

</br>





























































































































