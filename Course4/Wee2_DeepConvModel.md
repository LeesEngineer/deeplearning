</br>

# Why look at case studies

</br>

<p>Classic network:</p>

- LeNet-5

- AlexNet

- VGG

<p>ResNet also called residual network, trained a very deep 152 layer neural network.</p>

</br>

# Classic networks

</br>

## LeNet-5

</br>

<p>If you have a image that's 32 by 32 by 1, LeNet-5 can recognize handwritten digits. It has about sixty thousands parameters.</p>

<img width="2336" height="488" alt="QQ_1777019345974" src="https://github.com/user-attachments/assets/c67436ce-fb60-4767-9426-84c08f28cf0f" />

<p>LeNet-5 use sigmoid and tanh activation functions.</p>

</br>

## AlexNet

</br>

<p>AlexNet inputs start with 227 by 227 by 3 images. Using padding to implement "same convolution". It has sixty million parameters.</p>

<img width="2354" height="858" alt="QQ_1777020818514" src="https://github.com/user-attachments/assets/6e3a654b-28d1-4df9-ad12-16e10b35815e" />

<p>An aspect of AlexNet that made it much more better than LeNet was using ReLU activation function.</p>

</br>

## VGG-16

</br>

<p>Instead of having so many hyper parameters, it uses a much simpler network where you focus on just having conv layers. It really simplified these neural network architectures.</p>

`conv = 3 * 3 filter, s = 1, same    MAX-POOL = 2 * 2, s = 2`

<img width="2332" height="942" alt="QQ_1777029025123" src="https://github.com/user-attachments/assets/544e75c3-2e70-42ea-af72-49fe63b840d4" />

<p>It's a pretty large network. It has about 138 million parameters.</p>

</br>

# Residual Networks

</br>

<p>Very deep networks are difficult to train because of vanishing and exploding gradients types of problems.</p>

<p>Skip connection allows you to take the activation from one layer and suddenly feed it to another layer even much deeper in the neural network. Using that, you're going to build ResNets which enables you train very deep networks.</p>

<p>ResNets are built out of residual blocks.</p>

<img width="2352" height="388" alt="QQ_1777037651338" src="https://github.com/user-attachments/assets/fffd0641-5d55-4d03-a2fb-48b1b0b57030" />

<p>We take a[1] and just forward it, copy it much further into the neural network to here:</p>

<img width="2328" height="510" alt="QQ_1777038313485" src="https://github.com/user-attachments/assets/2d0cc694-e627-4710-bd09-857b2b887279" />

<p>And add a[1] <b>before applying to nonlinearity (ReLU)</b>. I inject it after the linear part but before the ReLU part. I call this shortcut. Rather than needing to follow the main path, the information from a[1] can now follow a shortcut to go much deeper into the network.</p>

<img width="740" height="326" alt="QQ_1777039220185" src="https://github.com/user-attachments/assets/84524ee2-7541-4d10-81c1-3e5fa98f1312" />

<p>The last equation becomes `a[l+2] = g(z[l+2] + a[l])`. The addition of this a[1] makes this a residual block.</p>

<p>The way you build a ResNet is by taking many of these residual blocks, and stacking them together to form a deep network.</p>

<img width="2158" height="344" alt="QQ_1777044209854" src="https://github.com/user-attachments/assets/4e5c89a0-6b71-4681-b650-2adcc64e42d4" />

<p>This picture shows five residual blcoks stacked together.</p>

<p>If you use standard optimization algorithm to train a <b>plain network</b>, you find that as you increase the number of layers the training error will tend to decrease, and after a while it'll tend to go back up.</p>

![Uploading QQ_1777044461614.png…]()

<p></p>





















































































































































































































































































