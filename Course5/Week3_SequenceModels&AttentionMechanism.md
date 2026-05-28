</br>

# Basic Models

</br>

<p>Sequence to sequence models are useful for everything from machine translation to speech recognition</p>

<p>Say you want to translate a French sentence into a English sentence.</p>

```
Jane visite I'Afrique en septembre.
-> Jane is visiting Africa in September.
From x<1>... x<5> up to y<1>... y<6>
```

<p>First let's have a network which we call it as <b>encoder network</b> be built as a RNN (GRU or LSTM). After you input the input sentence, then you can build a decoder network.</p>

<img width="1662" height="442" alt="QQ_1779895774255" src="https://github.com/user-attachments/assets/3b0e0aae-db8d-4e04-ab9a-5795ff26d16e" />

<p>It also works for image captioning, </p>

<img width="2504" height="772" alt="QQ_1779896255831" src="https://github.com/user-attachments/assets/348b7a2a-9825-4474-a56b-d0c3a42e1636" />

<p>In earlier course, we knew about how to learn the encoding of picture by maybe, say a pretrained AlexNet.</p>

<p>What we should do is getting rid of the softmax unit. Then we get a 4096-dimensional feature vector. You can feed it and feed it to RNN to generate the caption.</p>

<img width="1596" height="394" alt="QQ_1779952569729" src="https://github.com/user-attachments/assets/98e21c51-56c2-4ea6-9a14-f93b04b6a4bb" />

<p>It works pretty well, especially if the caption is not too long.</p>

</br>

# Picking the most likely sentence

</br>

<p>You can think of machine translation as building a conditional language model.</p>

<p>Language model output: P(y<1>, y<2>,... , y<T_y>)</p>

<p>Conditional language model outputs English translation conditions on some input French sentence.</p>

`P(y<1>, y<2>,... , y<T_y> | x<1>, x<2>,... , x<T_x>)`

<p>What you do not want is to <b>sample outputs at random</b>. If you sample words from this distribution--P(y | x), maybe you one time get a good translation. But maybe another time you get a different translation, which sounds a little awkward.</p>

<img width="2336" height="722" alt="QQ_1779957300362" src="https://github.com/user-attachments/assets/a54d2519-a55d-42dc-bfdf-b0083380acd1" />

<p>So we don't sample at rondom from this distribution. Instead, what you do is to find an English sentence y that maximizes that conditional probability.</p>

<p>Come up with an algorithm that can actually find the y that maximizes this term. The most common algorithm is called <b>beam search</b>.</p>

<p>We don't use greedy search? </p>

```
Jane is visiting Africa in September
Jane is going to be visiting Africa in September
```

<p>For example, if the algorithm has picked "Jane is" as the first two words, because the phrase "is going" is very common, probably the chance of "Jane is going" is higher than the chance of "Jane is visiting".</p>

<p>And it's impossible to rate them all. so we use a approximate search algorithm.</p>

</br>

# Beam Search

</br>

<p></p>






















































































































































































































