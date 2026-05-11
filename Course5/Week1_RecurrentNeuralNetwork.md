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

<p></p>




















































































































































































