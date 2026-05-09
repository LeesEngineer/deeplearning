<p>In order to implement neural style transfer, you need to look at the features extracted by ConvNets <b>at various layers, the shallow and the deeper layers of a ConvNet</b>.</p>

</br>

# What are deep ConvNets learning : Max Activation Visualization

</br>

<img width="2906" height="516" alt="QQ_1778120007504" src="https://github.com/user-attachments/assets/315d1581-4115-4431-be27-a397138f3a40" />

<p>Pick a unit in layer 1. Fine the nine image patches that maximize the unit's activation. Now plot which image patch maximize that unit's activation</p>

<img width="204" height="202" alt="QQ_1778120172864" src="https://github.com/user-attachments/assets/5f12afdc-677e-499c-998d-7951c48f7e5b" />

<p>It's looking for a edge or a line looks like this:</p>

<img width="142" height="138" alt="QQ_1778120200880" src="https://github.com/user-attachments/assets/34d16ebc-cbe1-4e2d-9369-5870d8628dfc" />

<p>Repeat for other units.</p>

<img width="2754" height="512" alt="QQ_1778121820586" src="https://github.com/user-attachments/assets/832dab5b-9e7d-466a-b68a-6170d71d1c5b" />

<p>Units in layer 1 often look for relatively simple features.</p>
 
<p>In deep layer, a hidden unit will see a larger region of image patches.</p>

<p>Layer 2 looks like it's detecting more complex shapes and patterns. A unit in layer 3 even starts to detect person.</p>

</br>

# Cost Function

</br>

<p>By minimize J, you can generate images you want.</p>

<img width="1156" height="1094" alt="QQ_1778122289504" src="https://github.com/user-attachments/assets/3c2ec22f-c030-45d2-96d2-aa22d0b712a1" />

`J(G) = alpha * J_content(C, G) + beta * J_style(S, G)`

<p>J_x measures how similar is the x image to the generate image. I think one hyperparameter could e enough.</p>

<hr>

<p>Now find the generated image G.</p>

- Initiate G randomly : G -> 100 by 100 by 3 (a moise image)

- Use gradient descent to minimize J(G) ; `G = G - dJ(G) / dG`

<img width="788" height="250" alt="QQ_1778123317097" src="https://github.com/user-attachments/assets/ec5f860c-3d53-4fcc-b529-9be60ba2a8ee" />

<img width="322" height="1078" alt="QQ_1778123330985" src="https://github.com/user-attachments/assets/8e94aa8b-93d7-4806-a40c-00920264ccac" />

</br>

## Content Cost Function

</br>

<p>Say you use hidden layer l to compute content cost. <b>If l is a very small number if you use layer 1, then algorithm would force your generate image's pixel values to be very similar to your content image. Whereas if you use a very deep layer, then it's just asking if there's a dog in your content image then make sure there's a dog somewhere in generated image.</b> layer l should be neither too shallow nor to deep</p>

<p>Then use a pre-trained ConvNet (e.g., AlexNet or VGG).</p>

<p>You want to measure how similar is the content image to the generated image.</p>

<p>Let a[l](c) and a[l](G) be the activation of layer l on the images. If a[l](c) and a[l](G) are similar, both images have similar content.</p>

`J_content(C, G) = || a[l](C) - a[l](G)||^2`

</br>

## Style Cost Function

</br>

<img width="2260" height="364" alt="QQ_1778157112365" src="https://github.com/user-attachments/assets/7f456446-e695-457e-bffc-27226d99a37f" />

<p>Say you area using layer l's activation to measure "style". We <b>define style as correlation between activations across channels.</b></p>

<img width="380" height="352" alt="QQ_1778307001945" src="https://github.com/user-attachments/assets/60a240dc-0756-41d6-b618-6326b80e63fd" />

<p>How to compute correlation between activations across different channels.</p>

<p>We think activations across two different channels as number of pairs. <b>Style is not the position of an object, but rather the statistical relationship between features.</b></p>

<p><b>And in CNN, a channel defines a feature detector.</b></p>

<p>So, we don't care about features, instead we care about the correlation between features.</p>

<img width="394" height="354" alt="QQ_1778309087339" src="https://github.com/user-attachments/assets/4df93ce9-52f1-4ebe-b422-3be68d243fcf" />

<img width="492" height="480" alt="QQ_1778309125423" src="https://github.com/user-attachments/assets/d52ed918-7f37-4cc4-bf42-d3cef7c41a22" />

<p>Say the red channel corresponds to the (0, 1) neuron, and try to figure out if there's this vertical texture. And the second channel corresponds to the (1, 0) neuron, which vaguely looking for orange colored patches.</p>

<p>If they are highly correlated what that means is whatever part of this image has the vertical texture, then that part will probably has this orange-ish tint. As for "uncorrelated", it's probably won't have that tint.</p>

<p>Correlation gives you one way to measure how often these different high level features occur together and don't occur together.</p>

<p>What you can do is measuring the degree of the first channel is correlated or uncorrelated with the other channels in generated image. Then you can measure how similar is the style of G to the style of S.</p>

</br>

## Style Matrix

</br>

<p>For G and S, you need to compute a style matrix.</p>

<p>Let a^[l]_{i, j, k} = activation at (i, j, k). And G^{[L](S)} is n[l]_c by n[l]_c.</p>

```
G^{[L](G)}_{kk'} = \sum_i^nh[l] \sum_j^nw[l] (a[l]_ijk * a[l]_ijk')
G^{[L](G)}_{kk'} = \sum_i^nh[l] \sum_j^nw[l] (a[l]_ijk * a[l]_ijk')
```

<p>This is an unnormalized cross-covariance, because we're not subtracting the mean.</p>

<p>G is nc by nc, in order to measure how correlated each <b>pair</b> of activations from different channels is.</p>

```
J[l]_style(S, G) = ||G[l](S) - G[l](G)||^2 / (2*nh * nw * nc)^2
J_style(S, G) = \sum_l lambda[l] J[l]_style(S, G)
```

<p>We use a additional set of parameters which we denote as lambda[l].</p>

<p>J allows you use different layers, both the earlier layers which measure relatively simpler low level features like edges, as well as some later layers which measure high level features.</p>

</br>

# 1D and 3D generalizations of models

</br>

<img width="1114" height="426" alt="QQ_1778317742748" src="https://github.com/user-attachments/assets/6f33bd5d-fc57-46d2-b047-56c11cdaab50" />

<p>For EKG, you may need this 5 dimensional filter.</p>

`14 conv 5 = 10`

<p>You can detect the different heartbeats in an EKG signal.</p>

<hr>

<p>Video needs 3D CNN. 2D CNN only display a signal frame. But timeing information is crucial. For example, a person is walking, a ball is moving. These are all required to understand the changes between consecutive frames.</p>

<p>CT scan also is 3D data, which gives a three-dimensional model of your body. What CT scan does is it takes different slices through your body.</p>

<img width="528" height="530" alt="QQ_1778320287397" src="https://github.com/user-attachments/assets/d15c9061-0435-4a81-86aa-813156d7251e" />


























































































































































































