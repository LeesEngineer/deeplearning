</br>

# Face Recognition

</br>

<p>People often talk about face verification and face recognition.</p>

<p>Verification: (one to one problem)</p>

- Input image, name/ID

- Output whether the input image is that of the claimed person

<p>Recognition: </p>

- Has a database of K persons

- Get an input image

- Output ID if the image is any of the K persons (or "not recognized")

<p>If you want an acceptable recognition error, you might actually need a verification system.</p>

<p>To building a face recognition system, you need to solve a one-shot learning problem.</p>

</br>

# One-shot learning

</br>

<p>For most face recognition applications, you need to rocognize a person given just one single image of the person's face. But deep learning algorithms don't work well if you only have one training example.</p>

<p>To make this work, you should learn similarity function: `d(img1, img2) = degree of difference between images`. If d(img1, img2) <= tau, same. </p>

</br>

# Siamese network

</br>

<p>A good way to achieve d function is to use a Siamese network</p>

<img width="2030" height="276" alt="QQ_1777815944819" src="https://github.com/user-attachments/assets/a1e24ab3-24ba-4875-94b7-58f3ffb5790e" />

<p>We input a image called x. Through a sequence of convolutional and pooling and FC layers, end up with a feature vector. Sometimes this is fed to a softmax unit to do classfication, but we're not going to use that. Instead we're going to focus on this vector called f(x). <b>We should think of f(x) as an encoding of the image.</b></p>

<p>Define the d as `d(x1, x2) = || f(x1) - f(x2)||^2`</p>

<p>The idea of running two identical convolutional neural networks on two different inputs and then comparing them calls a Siamese Network.</p>

<p><b>The parameters of NN define an encoding f(x). Learn parameters so that: If xi, xj are the same person, || f(xi) - f(xj) ||^2 is samll.</b> </p>

</br>

# Triplet Loss

</br>

<p>One way to get a good encoding of your image is to define and apply gradient descent on the triplet loss function. </p>

<p>To apply triplet loss, you need to compare pairs of images. You want their encodings to be similiar if they are the same person.</p>

<img width="604" height="414" alt="QQ_1777881877750" src="https://github.com/user-attachments/assets/e33b3312-58ee-45f9-80a2-af53b67c230a" />

<p>What you're going to do is always look at one anchor image. You want the distance between the anchor and a positive (positive meaning is the same person) image to be small. Whereas you want the distance between the anchor image and positive image to be much further.</p>

<p>We call these three image A, P, N. And we want `|| f(A) - f(P) ||^2 <= || f(A) - f(N) ||^2`</p>

`|| f(A) - f(P) ||^2 - || f(A) - f(N) ||^2 <= 0`

<p>One trivial way that is satisfied this expression is to learn everything equals zero. To make sure the nn doesn't output all zero: </p>

`|| f(A) - f(P) ||^2 - || f(A) - f(N) ||^2 + alpha <= 0`

<p>Alpha is called margin, and we want d(A, N) is much bigger than d(A, P).</p>

<p>Triplet loss function is defined on triples of images.</p>

`L(A, P, N) = max(0, || f(A) - f(P) ||^2 - || f(A) - f(N) ||^2 + alpha)`

`J = \sum_{i = 1}^M L(A^i, P^i, N^i)`

<p>Training set: 10k pictures (to generate triples) of 1k persons. Ten pictures on each person. If you only have 1 picture of a person, you can't actually train this system. After having training the system, you can apply it to your one-shot learning problem.</p>

<hr>

<p>Choosing the triplets A, P, N</p>

<p>During training, if A, P, N are chosen randomly, d(A, P) + alpha <= d(A, N) is easily satisfied. If A and N are two randomly chosen different persons, there's a very high chance. The neural network won't learn much from it.</p>

<p>So choose triplets that're "hard" to train on. Your algorithm tries to push d(A, N) up and push d(A, P) down.</p>

</br>

# Face verification and binary classification

</br>

<img width="2030" height="620" alt="QQ_1777891946574" src="https://github.com/user-attachments/assets/42ff5849-c755-41ba-9e29-9cfe6db9fb7e" />

<p>We treat face recognition just as a binary classification. This is an alternative to the triplet loss.</p>

`yhat = sigmoid(\sum_{k = 1}^128 w |f(xi)_k - f(xj)_k| + b)`

<p>xj or xi could be <b>precomputed</b>.</p>

<p>Treat fv as supervised learning.</p>

<p>Training set:</p>

<img width="844" height="580" alt="ab936208a1f94776161b2069ab26fecd" src="https://github.com/user-attachments/assets/39b889ca-5213-48b7-8db4-376504e5459c" />

<hr>








































































































































































































































































































































































































































