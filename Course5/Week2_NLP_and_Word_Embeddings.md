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

<p>Word2Vec is a more simple and computationally more efficient algorithm to learn embeddings.</p>

`I want a glass of orange juice to go along with my cereal.`

<p>In skip-gram model, what we're going to do is come up with a few context to target pairs to create supervised learning problem.</p>

<p>So rather than having the context be always the last four words or immediately before the target word, what I'm going to do is <b>randomly pick a word to be the context word</b>. Say we choose the word orange. What we can going to do is to pick another word within some window, say plus and minus five words of the context word. So might just by chance you pick juice to be a target word. Another pair could be orange and glass.</p>

```
Context         Target
orange          juice
orange          glass
orange          my
```

<p>So we set up a supervised learning problem where given the context word orange. You're asked to predict what is a randomly chosen word within, say a plus minus ten word window of that input context word. Obviously that's not a very easy learning problem.</p>

<p>But the goal of setting up this supervised learning problem is not to do well on this prolem itself, <b>we want to use this learning problem to learn good word embeddings.</b></p>

<hr>

<p>Say your vocabulary size is 10,000k. The basic supervised learning problem is that <b>we want to learn a mapping from context c to some target t</b></p>

`e_c = E * O_c`

<p>In the nn we formed, we take vector e_c and feed it to a softmax unit. The job of the softmax is to estimate probablities of different target words given the context word. Softmax outputs P(t | c)</p>

`P(t | c) = e^{theta^T_t * e_c} / (\sum^10,000k_{j = 1} e^{theta^T_j * e_c})`

`L(yhat, y) = - \sum y_i log yhat_i`

<p>Theta_t is associated with target word. Y is a one-hot vector. <b>Softmax layer has theta_t parameters</b></p>

`O_c -> E -> e_c -> [softmax] -> yhat`

<p><b>If you optimize this loss function, you actually get a pretty good embedding matrix or a set of embedding vectors.</b> And we learn theta_t.</p>

<p>This is called the skip-gram model.</p>

<p><b>During training, we used (orange, juice) and (orange, glass) pair, so it will continuously push e_orange to become a vector that easily matches with "juice", "glass" and other words in the "fruit" category. Consequently, the embeddings of orange, apple and banana will be very close.</b></p>

<hr>

<p>There are a couple problems.</p>

<p>The primary problem is computational speed. Every time you want to evaluate P(t | c), ni need to carry out a sum. One solution is to use a <b>hierarchical softmax</b> classfier, and what that means is that instead of trying to categorize something into 10,000k categories on one go, imagine you have one classifier that tells you is the target word in the first 5,000k words or is in the second 5,000k words in vocabulary. The second classifier tells you that this in the first 2,500k words, <b>and so on</b>.</p>

<p>You have a tree of classifier, so you don't need to sum over all 10,000k words.</p>

<p><b>In practice, the hierarchical softmax classifier can be developed so that the common words tend to be on top, whereas the less common words can be buried much deeper in the tree</b></p>

<img width="478" height="536" alt="QQ_1779612949853" src="https://github.com/user-attachments/assets/8cc0eec7-330e-44b2-be18-a807bc9be507" />

<hr>

<p>How to sample the context c?</p>

<p>One thing you could do is sampling uniformly at random from your training corpus. When you do that, there are some words like the, of, a, and, to and so on that appear extremely frequently. Whereas the other words like orange, apple and durian don't qppear that often. If you don't want you training set to be dominated by these extremely frequently occurring words. <b>Because then you spend almost all the effort updating e_c ofr those frequently occurring words. But you want to spend time updating the embedding for these less common words.</b></p>

<p>So we don't pick context word randomly, but instead there are different heuristics that you could use in order to balance out sampling from the common words to less common words.</p>

</br>

# Negative Sampling

</br>

<p>Say we define a new supervised learning problem. The problem is given a pair of words like orange and juice. We're going to predict is this a context-target pair? In this example, orange and juice was a positive example. Orange and king was a negative example, then we write 0 for this pair. We associate orange and juice with a label of 1 as the first row.</p>

<p>The positive example is generated exactly the same as how we generated in the skip-gram (sample a context word, look around a window of plus-minus 10 words to pick up a target word)</p>

<p>To generate a negative example, you use the same context word and pick up a word at random from the dictionary, and we label that as zero. <b>Under the assumption that if we pick a random word, it probably won't be associated with the context word.</b> </p>

<p>What we'll do is that, for some number times like k times, we're going to take the same context word and pick random words and label all those zero.</p>

<p>K is 5 to 20, and if you have a large data set then choose k to be smaller (2 to 5). We use k equals 4 here. </p>

<p>Then create a supervised learning problem where the algorithm inputs this pair of words as x and <b>it has to predict the target label</b>, and learning a mapping from x to y.</p>

<img width="866" height="980" alt="QQ_1779677515552" src="https://github.com/user-attachments/assets/40736285-b133-4a8b-aa4d-63793bd35bfc" />

<p>We define a logistic regression model, and you have a parameter vector theta for each possible target word.</p>

`P(y = 1 | c, t) = sigmoid(theta_t^T *e_c)`
 
<p>For every positive example you have k negative examples to train this logistic regression-like model.</p>

<img width="1304" height="472" alt="QQ_1779686685356" src="https://github.com/user-attachments/assets/22508e0e-ec1f-425b-bfba-87a6cad217a4" />

<p>So instead of having one giant 10,000k way softmax, we've instead turned it into 10,000k binary classification problems. On every iteration, we're only going to train five of them.</p>

</br>

## Select Negative Examples

</br>

<p>You could sample it according to the empirical frequency of words in your corpus, but you might end up with a very high representation of words like the, of, and, and so on.</p>

<p>One other extreme situation would be to say you use 1 over vocabulary_size to sample the negative examples uniformly at random. But <b>that's very non-representative of the distribution of English words</b>.</p>

<p>Therefore, we adopted a compromise approach.</p>

`P(w_i) = f(w_i)^{3/4} / \sum^10,000k f(w_j)^{3/4}`

<p>I'm not sure this is very theoretically justified.</p>

</br>

# GloVe Word Vectors

</br>

<p>GloVe stands for global vectors for word representation. Previously, we were sampling pairs context and target by picking two words that appear in close proximity to each other. What GloVe does is it starts off just by making that explicit.</p>

```
X_ij = # times i appears in the context of j
```

<p>I for t and j for c. Depending on the definition of c and t, x_ij equals x_ji.</p>

<p>X_ij is a count that captures how often do words i and j appear with each other, or close to each other. So GloVe model optimizes as following. We're going to minimize the difference between `(theta_i^T e_j - log X_ij)^2`</p>

<p>The term measures how far is the predicted relationship theta_i_T e_j from the target value log X_ij. We use gradient descent to minimize `\sum_{i = 1}^{10,000k} \sum_{j = 1}^{10,000k} (theta_i^T e_j - log X_ij)^2`</p>

<p>Log of zero is undefined, is negative infinity. So we add an extra weighting term f(x_ij). f(x_ij) = 0 if x_ij = 0.</p>

<p>Another role of f(X_ij) is that there are some words appear very often like the, a, of, is and so on, we don't give these words too much weight. And there are also some infrequent words like durion, <b>which you actually want to take into account, but not as frequently as the more common words.</b> So f(X_ij) can be a function that gives a meaningful amount of computation to less frequent words.</p>

<p>The roles of theta and e are completely symmetric (theta_i and e_j are symmetric), because <b>X_ij equals X_ji</b>. they play pretty much the same role and you could reverse them. So you can initialize theta and e both uniformly and use gradient descent to minimize them. When you're done for every word then take the average.</p>

`e_w^{final} = (e_w + theta_w) / 2`

<p>E_w and theta_w play symmetric roles unlike the previous model where theta and e play different roles.</p>

<hr>

<img width="1268" height="586" alt="QQ_1779722671413" src="https://github.com/user-attachments/assets/0f7d5da7-62d7-4af3-8029-384c0fc577f4" />

<p>You can not guarantee that the individual components of the embeddings are interpretable.</p>

`(A theta_i)^T (A e_j) = theta_i^T A^T A e_j = theta_i e_j`

<p>So if you plot a dimensional coordinate system, an axis could be a combination of gender, royal and age.</p>

</br>

# Sentiment Classification

</br>

<p>One of the chanllenges of sentiment classification is that you might don't have a huge labeld training set. But with word embeddings, you're able to build good sentiment classifier.</p>

<img width="2472" height="724" alt="QQ_1779840890725" src="https://github.com/user-attachments/assets/4fcbfde0-71e1-4b05-91ec-31a5bc217a78" />

<p>The first thing you can do is take their average and feed it to the softmax to predict five outputs.</p>

<p>This average work decently well. But it ignores word order. In particular, this is a negative comment: "Completely lacking in good taste, good service and good ambience.". The word good appears a lot. Your classifier might think it's a good review.</p>

<p>So you could use a <b>RNN</b></p>

<img width="2598" height="990" alt="QQ_1779842646849" src="https://github.com/user-attachments/assets/2de3ab00-20b7-44b4-8dcd-4b7a5f386958" />

<p>With an algorithm like this, it will be much better at taking word sequence into account.</p>

<p>And because your word embeddings are trained from a much larger data set, it works better.</p>

<p>"lacking in" to "ansent of"</p>

</br>

# Debiasing word embeddings

</br>

<p>We like to make sure that as much as possible that they're free of undesirable forms of bias</p>

<p>We have data-level debiasing, embedding debiasing, loss-function debiasing and so on. Such a danger is facing us mainly with word embeddings.</p>

<p>A learned embedding tends to output man is to computer programmer as woman is to homemaker and father is to doctor as mather is to nurse. It enforces a very unhealthy gender stereotype.</p>

<hr>

<p>Addressing bias in word embeddings</p>

<img width="888" height="968" alt="QQ_1779856418502" src="https://github.com/user-attachments/assets/f73402d9-784e-41d1-a533-5d19e793885d" />

<p>Some words are embedded as shown in the figure.</p>

<p>The first thing we're going to do is <b>identifying the direction corresponding to a particular bias we want to reduce or eliminate</b>. So how to identoify the direction corresponding to the bias. What can we do is taking the e_he and subtract the e_she. And take a few of these and average them</p>

```
e_he - e_she
e_male - e_female
...
```

<img width="1108" height="970" alt="QQ_1779864157242" src="https://github.com/user-attachments/assets/f38630bd-9f0f-4424-a8d7-3067320609dc" />

<p>What looks like is that this direction is gender direction. It might be 299-dimensional subspace. And this direction can be higher than 1-dimensional.</p>

<p>We're going to use a algorithm called a SVU (singular value decomposition).</p>

<p>The next step is a <b>neutralization step</b>. For every word that is undefinitional, project it to get rid of bias.</p>

<p>For words like doctor and babysitter, we project them onto this non-bias direction axis to reduce their component in the bias direction.</p>

<img width="1230" height="1098" alt="QQ_1779866556031" src="https://github.com/user-attachments/assets/a08c0f7d-adef-42c3-abbc-31c1c1c3dd18" />

<p>The final step is called equalization. In this step, we equalize pairs. You may have pairs of words like grandfather and grandmother, <b>where you want the only difference in their embedding to be the gender.</b></p>

<p>In the example, the distence or between babysitter and grandmother is actually smaller than the distence between babysitter and grandfather.</p>

<img width="1240" height="1126" alt="QQ_1779867075810" src="https://github.com/user-attachments/assets/19236e55-dd90-44b5-9ef5-0327821cce86" />

<p>What we'd like to do is to make sure that words like grandfather and grandmother are both the exactly same similarity from the word that should be gender neutral.</p>

<p>So we move grandmother and grandfather to a pair points that are equidistant form the axis.</p>

<p>The number of words that you should equalize is relatively small.</p>

































































































































































































































































