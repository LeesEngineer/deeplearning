<img width="2124" height="1334" alt="QQ_1777256094243" src="https://github.com/user-attachments/assets/f4105d40-0def-4300-b7d0-38a7a49c069a" /></br>

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

<p>If you use standard optimization algorithm to train a <b>plain network</b>, you find that as you increase the number of layers the training error will tend to decrease, and after a while it'll tend to go back up. And in theory, as you make a neural network, it should do better and better. But in reality, Using optimization algorithm to train a deep network without ResNet has a much harder time training.</p>

<img width="756" height="674" alt="QQ_1777098875445" src="https://github.com/user-attachments/assets/27a32d77-b733-4b23-a3aa-11c58c8a745d" />

<p>What happens with ResNets is that even as the number of layers gets deeper, you can have the performance of the training error keep on going down.</p>

<img width="776" height="668" alt="QQ_1777102136997" src="https://github.com/user-attachments/assets/006b562c-f8c0-4a61-93cf-6b1bd54d32a0" />

</br>

# Why ResNets work

</br>

<p>If you have X feeding in to a big neural network and it outputs a[l]. </p>

<img width="900" height="192" alt="QQ_1777104049078" src="https://github.com/user-attachments/assets/533ca159-cc38-4233-8ca5-cbb3ed910b15" />

<p>Now you add a couple extra layers to it, and make these two layers a ResNet block with shortcut.</p>

```
a[l+2] = g(z[l+2] + a[l])
a[l+2] = g(w[l+2] * a[l+1] + b[l+2] + a[l])
```

<p>Notice that if you are using L2 regularisation or weight decay, that would tend to shrink the value of w[l+2], and w is key term to here.</p>

<p>If w[l+2] is equal to zero, and for the sake of argument b is also equal to 0. then a[l+2] = ReLU(a[l]) = a[l].</p>

<p>The identity function is easy for residual block to learn. Which means that adding these two layers in nn, it doesn't hurt your nn ability to do as well as plain nn. It's easy to get `a[l+2] = a[l]` because of skip connection.</p>

<p>One more detail in ResNets that's worth discussing, which is we're assuming that z[l+2] and a[l] have the same dimension. So conv nn uses a lot of same convolution.(same padding). If the dimension are not equal, then add a matrix w_s before a[l]. It could be a fixed matrix that just implements zero padding.</p>

<img width="2170" height="382" alt="QQ_1777107374164" src="https://github.com/user-attachments/assets/c24479b1-eba0-4d52-8f31-43da943b861b" />

<img width="2200" height="400" alt="QQ_1777107481554" src="https://github.com/user-attachments/assets/b30bcc69-1c36-4a4c-9a82-654c5302b46e" />

</br>

# Network in Network and 1 by 1 convolutions

</br>

<img width="1942" height="466" alt="QQ_1777125252423" src="https://github.com/user-attachments/assets/13631508-3975-477d-9f77-11e8f9411ae6" />

<p>You just multiply the input by some number, but that's the case of one channel images. If you have a 6 by 6 by 32, then a convolution with a 1 by 1 filter can make it better.</p>

<img width="1986" height="508" alt="QQ_1777126074911" src="https://github.com/user-attachments/assets/fc6dd745-76ba-4705-861e-f2d66d9ae354" />

<p>You can think the 32 numbers you have in this as a 1 by 1 by 32 vector, and you have one neuron that's taking 32 numbers as input. It performs linear computation <b>along the channel dimension.</b> "<b>Create a fully connected layer for each pixel</b>"</p>

<p>More generally, if you have not just one filter as if you have not just one unit.</p>

<p>You can think about the 1 by 1 convolution is that it has a fully connected layer that applies to each of the 32 different positions. The FC inputs 32 numbers and outputs number of filters. And doing this at each of the 32 positions.</p>

<img width="1324" height="540" alt="QQ_1777172040794" src="https://github.com/user-attachments/assets/0c745b43-ed57-4124-9dd3-4eef4e4ac65c" />

<p>We can use pooling layers to shrink the height and width and use 1 by 1 convoluton to shrink the number of channels.</p>

</br>

# Inception

</br>

<p>When designing a conv layer, you have to pick do you want 3 by 3 or 5 by 5 filter, or do you want to have a pooling layer. Inception layer will make the decision for you, and this makes the network architecture more complicated but it works remarkably well.</p>

<p>We use same convolution and use padding for max pooling to make all the dimensions match.</p>

<img width="2004" height="850" alt="QQ_1777193132747" src="https://github.com/user-attachments/assets/b7bfde1d-e81d-46ce-b91b-0924d4ffd0d4" />

<p>With a inception module like this, you can input some value, and output this which adds up all these numbers, 32 + 32 + 128 + 64 = 256. </p>

<p>This basic idea is that instead of you needing to pick one of these filter size or pooling, you can do them all and concatenate the results.</p>

<hr>

<p>Using 1 by 1 convolution to create the bottleneck layer to reducing the computational cost significantly. As long as you implement this bottleneck layer reasonable, <b>you can shrink down the representation size significantly and doesn't hurt the performance.</b></p>

<img width="1390" height="660" alt="QQ_1777194041687" src="https://github.com/user-attachments/assets/f7838968-abaa-4491-b02a-d2b1fb95af82" />

<p>Let's compute the computational cost of 28 by 28 by 32: `(28 * 28 * 32) * (5 * 5 * 192)`, it's about 120 million.</p>

<b>Using 1 by 1 convolution.</b>

<img width="2104" height="644" alt="QQ_1777211729926" src="https://github.com/user-attachments/assets/35bac0d0-c4ea-48fa-88c2-3dd36a9f0a91" />

<p>It's called a bottleneck layer. We shrink the representation before increasing the size agagin. The cost is: `(28 * 28 * 16 ) * (192) + (28 * 28 * 32) * (5 * 5 * 16) = 2.4 M + 10.0 = 12.4 M`</p>

<p>You reduce the computational cost from about 120 million down to 12.4 million.</p>

</br>

# Inception module

</br>

<img width="2444" height="1322" alt="QQ_1777256167902" src="https://github.com/user-attachments/assets/38e035a2-4bc6-41ea-be15-76020f1dfb2d" />

<p>This is one inception module.</p>

<img width="2448" height="544" alt="QQ_1777256303298" src="https://github.com/user-attachments/assets/7cc42426-288d-4fd6-b2cf-ccd8d6d34bf0" />

<p>This is GoogleNet. To summarize, if you understand the inception module, then you understand the inception network, which is the inception module repeated a bunch of times throughout the network.</p>

</br>

# Transfer Learning

</br>

<p>You can download open-source weight, and use that as a good initialization for youe own network, then use transfer learning to transfer knowladge from very large public data sets to your own problem.</p>

<p>Let's create a classifier to classify cat, your cats are called Tigger, Misty or neither. <b>You probably don't have a lot of pictures of Tigger or Misty. Your training set would be small.</b> You can download a open source implementation of a neural network and it's weights.</p>

<img width="2522" height="312" alt="QQ_1777261172820" src="https://github.com/user-attachments/assets/e1d3491e-ce22-4ea7-8d07-b5e902aac5e0" />

<p>ImageNet has 1000 different classes. What you can do is that get rid of its softmax layer and create your own softmax layer that outputs Tigger, Misty or neither. <b>Don't forget to freeze the parameters in all of these layers in network.</b> You would just train the parameters associated with your softmax. </p>

<p>If you have a lot of pictures, you should <b>freeze fewer layers, and train these later layers,</b> or you can also blow away these last few layers and use your own new hidden units.</p>

<img width="2212" height="412" alt="QQ_1777275355116" src="https://github.com/user-attachments/assets/3c2e9d65-1008-44e5-9742-fb71d9112d5b" />

<p>One pattern is that if you have more data, the number of layers you freeze could be smaller, and the number of layers you train on the top could be greater.</p>

<p>Transfer learning is just very worth considering usless you have an exceptionally large data set and very large computational budget.</p>

</br>

# Data augmentation

</br>

<p>The simplest data augmentation method is mirroring on the vertical axis. Another commonly used technique is <b>random cropping</b>.</p>

<p>The third method is color shifting.</p>

<img width="1602" height="1056" alt="QQ_1777279785066" src="https://github.com/user-attachments/assets/18ffec18-931a-4b69-9461-2a1f4d756fa2" />

<img width="2070" height="892" alt="QQ_1777280526824" src="https://github.com/user-attachments/assets/4ffb0cd9-e94f-4071-83b6-d8177028cfa7" />

























































































































































































































