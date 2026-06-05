</br>

# Transformer Network Intuition

</br>

<p>RNN, GRU, LSTM, these are all sequential models. They ingested inputs. Each unit was like bottleneck to the flow of information, because to compute the output of the final unit, you first have to compute the outputs of all of the units that come before.</p>

<p>Transformer allows you to run a lot more of these computations for an entire sequence in parallel. So you can ingest an entire sentence all at the same time rather than just processing it one word at a time from left to right.</p>

<p>The major innovation of the transformer architecture is combining the use of attention based representations and a CNN style of processing.</p>

<img width="2288" height="634" alt="QQ_1780642098313" src="https://github.com/user-attachments/assets/a377fb95-b4d9-4648-a2e9-55d64dd97528" />

<p>RNN sequentially generates y. And CNN computes pixels in parallel.</p>

<p>There will be two key ideas:</p>

- Self-Attention

- Multi-Head Attention

<p>Multi-Head attention is basic a for-loop over the self-attention process.</p>

</br>

# Self-Attention

</br>

<p>To use attention with a style more like CNN, you need to calculate self-attention, where you create attention-based representations for each of the words in your input. </p>

<p>A(q, K, V) = attention-based vector representation of a word.</p>

<p>Self-attention has some similarity with RNN attention, it also involves a softmax.</p>

<img width="1050" height="192" alt="8e106e8b0fdc83c735ceab332533153f" src="https://github.com/user-attachments/assets/0e880ec8-0508-4976-8f37-9a1f309abb78" />

<img width="884" height="216" alt="4e39ad34d9c378e55fa927c689aca63e" src="https://github.com/user-attachments/assets/00b5e8a4-760f-41cc-b497-20cee13c4e0f" />

<p>You can think of `q * k<i>` as being akin to attention values. The difference is that, for every word you have three values called the query, key and value.</p>

<hr>

`Jane visite I'Afrique en septembre.`

<p>Let's see how to compute A<3>.</p>

<p>If x<3> is the <b>word embedding</b> for 'I'Afrique', the way that q<3> is computed is a learned matrix, and so on.</p>

```
q<3> = w^Q * x<3>
k<3> = w^K * x<3>
v<3> = w^V * x<3>
```

<p>Query represents a question: When computing A<3>, what's happening there. The key looks at all of the other words and by the similarity to the query, helps you figure out which word gives the most relevant answer.</p>

<p>So we're going to compute the inner product between q<3> and every k. <b>It tells us how good is x<i> an answer to the question of what's happening in x<3></b>.</p>

<p>This operation pulls up the most information that's needed to help us to compute the most useful representation A<3>.</p>

<p>Just for intuition building, if k<1> represents that this word is a person, and k<2> represents that this is an action, then you may find that q<3> inner product with k<2> has the largest value. This might suggest that <b>visite gives you the most relevant context for what's happening in Africa</b>, which is that, it's viewed as a destination for a visit.</p>

<img width="1976" height="860" alt="QQ_1780646427478" src="https://github.com/user-attachments/assets/1ab0f074-3ad4-4947-ae98-242c29bb11fa" />

`A<3> = A(q<3>, K, V)`

<p>For sentence "I love AI", to compute A<3>:</p>

`A(q_{AI},K,V) = 0.001v_{I} + 0.992v_{love} + 0.007v_{AI}`

<p><b>The key adventage of this representation is that the word 'I'Afrique' isn't some fixed word embedding, instead, it lets the self-attention mechanism realize that 'I'Afrique' is the destination of a visite. And thus compute a richer representation for this word</b>.</p>

<img width="918" height="156" alt="QQ_1780646785293" src="https://github.com/user-attachments/assets/01bb024a-5e46-41c6-85a2-f6a56bfe497f" />

<p>This is a vectorized representation of the upper equation. The term in the denominator is to scale the dot-product. Because the size of this inner product increases as the dimension d_k increases.</p>

<p>If d_k is 2, these inner product could be 2, 3 and 4. After softmax, they will be 0.09, 0.24, 0.67. But if d_k is 512, inner products might be 50, 70, 90. And after softmax, they will be 0, 0 and 1. <b>This leads to softmax saturation, where almost all the weights are concentrated in one position</b>. So it doesn't explode.</p>

</br>

# Multi-Head Attention

</br>

<p>It's just a big for-loop over the self-attention mechanism</p>










































































































































































































