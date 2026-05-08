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

<p></p>

















































































































































































































