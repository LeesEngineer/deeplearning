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

<p>The first thing Beam search has to do is trying to pick the first word of the translation.</p>

<img width="1030" height="476" alt="QQ_1780133788899" src="https://github.com/user-attachments/assets/d906781d-ad7e-4732-a4b9-6c05008e0818" />

<p>We use it to evaluate P(y<1> | x). Greedy search try to pick the most likely words and move on. Beam search can instead consider multiple alternatives. So it has a parameter called B (Beam Width). Maybe it finds that the choices in jane september are the most likely threepossibilities for the first word</p>

<p>Then what beam search will do is for each of these three choices condsider what should be the second word. It evaluates P(y<1> | x, y<2>).</p>

<img width="1476" height="364" alt="QQ_1780210225974" src="https://github.com/user-attachments/assets/5e318e29-900a-49e6-a21c-cfd5976ba8e2" />

<p>What we ultimately care about in second step is to find the pair of (yhat<1>, yhat<2>) that is most likely.</p>

`P(y<1>, y<2> | x) = P(y<1> | x) * P(y<2> | x, y<1>)`

<p>You have seen how you can evaluate the probability of the second word.</p>

<p>Because B equals three, you get 30,000k probabilities, and <b>pick the top three.</b></p>

<p>The third step.</p>

<img width="1372" height="334" alt="QQ_1780210857860" src="https://github.com/user-attachments/assets/1d311744-c32e-4ad8-877d-2e6e29de7945" />

<p>It allows you to evaluate P(y<3> | x, y<1>, y<2>).</p>

`P(y<1>, y<2>, y<3> | x) = P(y<1> | x) * P(y<2> | x, y<1>) * P(y<3> | x, y<1>, y<2>)`

</br>

# Refinements to beam search

</br>

## Length normalization

</br>

<p>beam search is trying to maximizing this probability. </p>

`\prod_{t = 1}^{T_y} P(y<t> | x, y<1>, ..., y<t-1>) = P(y<1> ... y<T_y> | x)`

<p>If you're implementing this, these probabilities are all numbers less than 1, often they're much less than 1. Multiplying a lot of numbers less than 1 will result a very tiny number <b>which can result in numerical underflow.</b></p>

<p>So instead of maximizing this product, we will take logs.</p>

`\sum_{y = 1}^{T_y} log P(y<t> | x, y<1>, ..., y<t-1>)`

<p>We get a more numerically stable algorithm that is less prone to rounding errors.</p>

<hr>

<p>There is other change to this objective function. Because you're multiplying a lot of terms that are less than 1 to estimate the probability. So this objective function has a undesirable effect that <b>maybe it unnaturally tends to prefer very translations.</b> The same thing is true for log.</p>

<p>We could normalize this by dividing it by T_y.</p>

<p>And we have a softer approach, we have T_y to the power of alpha, maybe alpha is equal to 0.7 that is somewhat in between full normalization and no mormalization.</p>

<p>Using alpha is a heuristic.</p>

</br>

## Beam width

</br>

<p>The larger B is, the more possibility you're considering. Ten is common in applications.</p>

<p>It's not uncommon to see people use beam widths of 1,000 or 3,000, but it's domain dependent.</p>

</br>

# Error analysis on beam search

</br>




































































































































































