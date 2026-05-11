<img width="1110" height="694" alt="QQ_1778499790032" src="https://github.com/user-attachments/assets/8f42e259-0095-4134-a8a1-b4827944f5a5" /><img width="1140" height="750" alt="QQ_1778499769526" src="https://github.com/user-attachments/assets/74e25eb8-f23c-44ab-872e-a4035a74eb10" /><img width="2576" height="1012" alt="QQ_1778498322340" src="https://github.com/user-attachments/assets/a6bd4aa9-115a-463c-89f5-1b63d342da30" /><img width="2304" height="1080" alt="QQ_1778326775514" src="https://github.com/user-attachments/assets/89dcb5de-5329-472d-bd8f-411efe570446" />

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


























































































































































