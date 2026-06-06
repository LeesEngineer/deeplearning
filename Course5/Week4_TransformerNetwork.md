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

<p>It's just a big for-loop over the self-attention mechanism. Every head is a set of independent attention (W_i^Q, W_i^K, W_i^V). For example, head 1 focuses on syntactic relationships and head 2 focuses on positional relationship.</p>

<img width="2750" height="740" alt="QQ_1780651102022" src="https://github.com/user-attachments/assets/c88bd6e2-dcf1-4097-884d-ec3f9da67eda" />

<p>In previous video we compute A<1> to A<5>. But now we do this a handful of times. We might now have eight heads. For head 2 we have a new set of matrixs. Maybe the first question is what's happening (<b>visit</b>), and the second question is when is something happening. In this case, maybe the inner product between the septembre key and I'Afrique query will be this largest.</p>

<img width="2784" height="770" alt="QQ_1780651727216" src="https://github.com/user-attachments/assets/6b98e89f-56fe-45a4-90ee-6a017e1a705b" />

<p>Then the concatenation of these eight values is used to compute the output of the <b>multi-head attention.</b></p>

`MultiHead(Q, K, V) = concat(head_1, head_2, ..., head_h) W_o`

<p>You can actually output all the heads in parallel.</p>

</br>

# Transformer Network

</br>

<p>Let's put it all together.</p>

<p>It's useful to add <SOS> and <EOS> tokens in sequence translation tasks.</p>

<p>The first step is these embeddings get fed into an encoded block, which has a multi-head attention layer. And feed in Q, K and V.</p>

<p>This layer then produces a matrix that can be passed into a <b>Feed Forward Neural Network</b>, which helps determines <b>what interesting features there are in the sentence</b>. </p>

<p>This encoding block is repeated N times (A typical value for N is six). </p>

<img width="1066" height="1146" alt="QQ_1780653261854" src="https://github.com/user-attachments/assets/e670966e-b979-42bf-9621-c18f9267a48b" />

<p>After six times through this block, then feed the output of the encoder into a decoder block.</p>

<p>The first ouput will be <SOS></p>

<p>At every step, the decoder block will input the first few words whatever we've already generated of the translation.</p>

<p>So <SOS> gets fed into this Multi-Head Attention block. <SOS> is used to compute Q, K and V for this multi-head attention block.</p>

<p><b>This first block's output is used to generate the Q matrix for the next multi-head attention block and the output of the encoder is used to generate K and V.</b></p>

<img width="2802" height="1118" alt="QQ_1780654122196" src="https://github.com/user-attachments/assets/b23148a9-f8ef-4515-ab19-9a2dc68af6fa" />

<p>Why is it structured this way: The inputs down here is what you have translated of the sentence so far. This will ask query to say what's the start of the sentence. And then we will pull comtexts from K and V which are translated from the French sentence <b>to try to decide what is the next word to generate</b>.</p>

<p>To finish the description of the decoder block, the multi-head attention block outputs the value and feed it to a <b>Feed Forward Neural Network</b>. </p>

<b>The decoder is going to be also repeated N times.</b>

<img width="2908" height="1126" alt="QQ_1780654567651" src="https://github.com/user-attachments/assets/c3b72f26-733c-4e8f-8be6-487ae0b4aba4" />

<p>The job of this feed forward nn is to predict the next word.</p>

<p>Then feed Jane into the inputs as well (<SOS> Jane). Now the next query comes from `<SOS> Jane`</p>

<hr>

<p>There're a few extra bells and whistles to transformer.</p>

<p>The first is <b>positional encoding</b> of the input. If you recall the self-attention equation, there's nothing that indicates the position of a word.</p>

<p>The way you encode the position of elements in the inputs is using combination of sine and cosine</p>

<img width="462" height="234" alt="QQ_1780672719825" src="https://github.com/user-attachments/assets/307fec65-8d21-41bd-9a89-379800b3fad5" />

<p>d_model is the dimension of word embedding. We are going to create a positional embedding vector of the same dimension. For word Jane, pos is equal to 1, and i refers to the different dimensions of the vector</p>

<img width="578" height="444" alt="QQ_1780673781961" src="https://github.com/user-attachments/assets/835c62ed-7a0c-48c1-a661-4edb4e974c2c" />

<p>What's the position encoding does with sine and cosine is <b>create a unique positional encoding vector</b>.</p>

<img width="1072" height="838" alt="QQ_1780726341747" src="https://github.com/user-attachments/assets/3f5b063c-1a5e-47c7-af49-76571385742b" />

<p>When i equals 2, you will end up with a <b>lower frequency</b> sinusoid.</p>

<p>Then the position encoding P1 is added directly to x<1>. So each of the word vector is also influenced with where in the sentence the word appears.</p>

`X = E + PE`

<hr>

<p>In addition to adding this position encoding to word embedding, <b>you'd also pass them through the network with residual connections.</b> They are similar to ResNet</p>

<p><b>Their purpose is to pass along positional information through the entire architecture.</b></p>

<p>Without residual connections:</p>

```
H_1 = Attention(X)
H_2 = Attention(H_1)
H_3 = Attention(H_2)
```

<p>After many layers, for example, the initial position information may be gradually modified or even lost. So we use:</p>

```
H_1 = X + Attention(X)
H_2 = H_1 + Attention(H_1)
H_3 = H_2 + Attention(H_2)
```

<hr>

<img width="764" height="928" alt="QQ_1780726887310" src="https://github.com/user-attachments/assets/91101864-ad74-41da-b519-7c76919d3c6a" />

<p>The transformer also uses a layer called <b>Add & Norm</b> that is very similar to the <b>Batch Norm layer</b> that helps speed up learning. It does residual connection and <b>layer normalization</b>.</p>

```
H_1 = LayerNorm(X + Attention(X))
H_2 = LayerNorm(H_1 + Attention(H_1))
H_3 = LayerNorm(H_2 + Attention(H_2))
```

<p>And this add&norm layer repeats throughout this architecture.</p>

<img width="2820" height="1306" alt="QQ_1780728137497" src="https://github.com/user-attachments/assets/7ea7bfbf-27ce-47cf-aca7-e4e048211ec9" />

<p>Finally, for the output of decoder block, there's also a linear layer and a softmax layer.</p>

</hr>

<p>We also have a masked multi-head attention in decoder block. It's important only during the training process where you're using a data set of correct French to English translations to train your transformation.</p>

<p>Masked multi-head attention is used to make sure that model can only see words that have been generated, and <b>cannot peek at future words.</b></p>

<p>What does multi-head attention do is that:</p>

`Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V`

<p>When it computes attention score for one token, it uses all of the token. But in decoder, we generate words one by one.</p>

<p>Although when training the <b>decoder</b>, you can see the whole sentence. But <b>model need to learn how to predict the t-th word only with the previous t-1 words</b></p>

<p>What mask does is blocking out the last part of sentence</p>

<p>For sentence "I love machine learning". The initial attention score is:</p>

```
QK^T =
\begin{bmatrix}
1 & 2 & 3 & 4 \\
1 & 2 & 3 & 4 \\
1 & 2 & 3 & 4 \\
1 & 2 & 3 & 4 \\
\end{bmatrix}
```

<p>After adding mask, it becomes:</p>

```
\begin{bmatrix}
0 & -\infty & -\infty & -\infty \\
0 & 0       & -\infty & -\infty \\
0 & 0       & 0       & -\infty \\
0 & 0       & 0       & 0       \\
\end{bmatrix}
```

`Attention = softmax\left(\frac{QK^T + Mask}{\sqrt{d_k}}\right)V`

<img width="758" height="1212" alt="QQ_1780728560849" src="https://github.com/user-attachments/assets/153ab8dc-2ef8-47ca-8f0a-259482730028" />























































































































