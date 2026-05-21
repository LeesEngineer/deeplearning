</br>

# Word Representation

</br>

<p>Word embedding is a way of representing words that let your algorithms automatically understand analogies like man is to woman as king is to queen (king - man + woman = queen).</p>

<p>Through these ideas of word embeddings, you are able to build NLP applications <b>even with relatively samll label training sets</b>.</p>

<p>So far, we've been representing words using a vocabulary and one-hot vector.</p>

<p>One of the weeknesses of this representation is that it treats each word as a thing onto itself. <b>This doesn't allow the algorithm easily generalize the cross words.</b></p>

<p>Say you have two sentence: "I want a glass of orange ____" and "I want a glass of apple ____". The nn might know the blank is (orange) juice. But it doesn't know the relationship between apple and orange is closer than the relationship between any of the other words (man, woman) and orange.</p>

<p>It's not easy for the learning algorithm to <b>generalize from knowing that orange juice is a popular thing to recognize that apple juice also be a popular thing.</b></p>

<p><b>This is because the inner product between any two different one-hot vectors is zero.</b> So it doesn't know apple and orange are much more similar than orange and king. </p>

<p>We can use a featurized representation. We could learn a set of features and values for each of them.</p>

<img width="2252" height="954" alt="QQ_1779203084264" src="https://github.com/user-attachments/assets/9117868c-4ece-4af6-aa56-e376aa78ebc4" />

<p>Say you have three hundards of dimensions. Many values of orange and apple is similar. So the learning algorithm that has figured out that orange juice is a thing can also figure out that apple juice is a thing.</p>

<hr>

<p>We can take this 300 dimensional data and <b>embed it in a two dimensional space</b> to visualize it.</p>

<img width="946" height="720" alt="QQ_1779266641414" src="https://github.com/user-attachments/assets/cae74f69-4ab7-4dbb-b6f6-cdf8df4fc89f" />

<p>The more they have in common, the closer they are.</p>

<p>You can think of a 300 dimensional space, but I can't draw it, so I use 3D instead.</p>

<p>Say orange gets embedded to a point, and apple gets embedded to here.</p>

<img width="522" height="526" alt="QQ_1779266979897" src="https://github.com/user-attachments/assets/f7a436f6-fe22-49f3-bc3b-da82fd5f2a51" />

</br>

# Using Word Embeddings

</br>

<p>Continue our name entity rocognition example.</p>

<img width="2108" height="576" alt="QQ_1779267660308" src="https://github.com/user-attachments/assets/58a70e0f-db56-40db-83c2-dd7735461353" />

<p> After having trained a model that uses word embeddings as inputs, say you have another sentence "Robert Lin is an apple farmer.". Knowing that orange and apple are similar will make it easier to generalize to figure out that Pobert Lin is also a human.</p>

<p>What if you see much less common words in your training set, such as durian cultivator, you might not seen these word in your training set. But if you have learned a word embedding. Algorithm knows that orange is a kind of fruit and cultivator is a kind of farmer, you might generalize from orange farmer in training set to know that durian cultivator is also probably a person.</p>

<p>By examining tons of unlabeled text, you can figure out that orange and apple are similar, then groups them together. </p>

<p>Transfer learning and word embeddings: </p>

1. Learn word embeddings from very large (1 - 100 Billion words) text corpus, or you can download pre-trained embedding

2. Transfer embedding to new task with smaller labeled training set. (say, 100k words)

3. Optional: Continue to finetune the word embeddings with new date (Only if task 2 has a pretty big data)

<p>If you want to transfer from task A to task B, the process of transfer learning is just most useful when you have a ton of data for A and relatively smaller data for B.</p>

<p>We have a fixed vocabulary of, say, 10,000 words, and we will learn 10,000 fixed encodings (embedding) for each of the words.  </p>

<p>Using embedding vectors <b>allows your algorithm to generalize much better, or you can learn from less label data.</b></p>

<hr>

<p>Word embedding have a interesting relationship to the face encoding.</p>

<img width="2222" height="770" alt="QQ_1779282225184" src="https://github.com/user-attachments/assets/a955a032-9328-45cb-9e42-a2f6015a417f" />

<p>We have 128 dimensional representations for different faces, and compare these encodings.</p>

<p>The terms of encoding and embedding are used somewhat interchangeably. </p>

</br>

# Properties of word embeddings

</br>

<p>Man is to woman and king is to queen. Is it possible to have a algorithm to figure this out automatically?</p>

<img width="2134" height="620" alt="QQ_1779348322157" src="https://github.com/user-attachments/assets/b6fe3778-c3bf-4480-b50a-100b8e0fbb6d" />

<p>We have e_man and e_woman. And one interesting property of these vectors is that if you take the vector, e_man substract e_woman, then you end up with approximately e_king substract e_queen.</p>

<p>Let's turn this into algorithm.</p>

<p>The word embeddings live in maybe a 300 dimensional space. The vector difference between man and woman is very similar to the vector difference between king and queen.</p>

<img width="882" height="746" alt="QQ_1779350269309" src="https://github.com/user-attachments/assets/b76e1cb9-95b2-4fa3-8ccd-71be1d46c05f" />

<p>So in order to carry out this kind of analogical reasoning to figure out man is to woman is king is to what.</p>

`e_man - e_woman \approx e_king - e_what`

<p>To find this word w, use similarity function and max `Sim(e_w, e_king - e_man + e_woman)`</p>

<p>If you learn a set of word embeddings and find the word w that maximizes the similarity, you can actually get the right answer.</p>

<p>The most commonly used similarity function is called <b>cosin similarity</b>.</p>

`sim(u, v) = u^T v / (|| u || * || v ||)`

<p>If u and v are similar, the inner product will be large.</p>

</br>

# Embedding Matrix

</br>

<p>When you are implement the word embedding learning algorithm, you actually end up with a embedding matrix.</p>

<img width="1796" height="726" alt="QQ_1779352661207" src="https://github.com/user-attachments/assets/860fd99e-eb06-41e4-907d-ddcdd8bd5552" />

<p>We call this matrix as E. Say the word orange is at position 6087 and O_6087 is it's one-hot vector. The result of "E * O_6087" is e_6087 which is 300 by 1. But in practice, we use a special function to look up a embedding vector.</p>

</br>

# Learning Word Embeddings

</br>

<img width="1948" height="206" alt="QQ_1779354743101" src="https://github.com/user-attachments/assets/154b2193-a87d-444a-bfd5-8897b9502439" />

<p>Let's build a nn to predict the next word.</p>

<img width="1536" height="756" alt="QQ_1779354916313" src="https://github.com/user-attachments/assets/a8abd10e-cf31-462b-b75a-abd610b1d2f7" />

<p>You get a bunch of embedding vectors by E * O. We <b>fill all of them into a nn hidden layer, then this hidden layer feeds to a softmax. The softmax layer classifies among the 10,000 possible outputs to predict the final word.</b> If your training set has word juice, it would predict juice.</p>

<img width="1176" height="910" alt="QQ_1779355302888" src="https://github.com/user-attachments/assets/b0d49461-6e6b-4985-8ce1-59b91258fb66" />

<p>We stack these six vector together, so the input would be a 1,800 (300*6) dimensional vector.</p>

<p>What's more commonly done is to <b>have a fixed historical window</b>. You might always want to predict the next word given say the previous four words, and get rid of 'I' and 'want'. So you input a 1,200 dimensional vector.</p>

<p><b>Using a fixed history, means that you can deal with arbitrarily long sentences.</b> This algorithm will learn pretty good word embeddings.</p>

<p>Our orange and apple juice problem is in the algorithm's incentive to learn similar word embeddings for orange and apple.</p>

<hr>

<p>Now we use a more complex sentence to illustrate this algorithm.</p>

<p>If your goal isn't to learn the previous language model, you can choose other context, for example, you can pose a learning problem where the context is four words on the left and right. Then feed into a nn.</p>

```
I want a glass of orange juice to go along with my cereal.
target: juice
context: Last 4 words
         4 words on left & rightL: a glass of orange ____ to go along with
         Last 1 word
         Nearby 1 word: Skip-Gram 
```

</br>

# Word2Vec

</br>











































































































































































































































































































































































































