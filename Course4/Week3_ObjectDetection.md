</br>

# Object localization

</br>

<p>We are already familiar with the image classification. The problem you learn to build a network to address later is classification with localization. We are trying to recognize and localize.</p>

<p>In detection problem, there can be multiple objects of different categories within a single picture. So the idea you learned about for image classification will be useful for classification with localization. And then the ideas you learn for localization is useful for detection.</p>

<img width="2228" height="524" alt="QQ_1777296940257" src="https://github.com/user-attachments/assets/aabff093-a03b-4d70-bc9b-896ce6061da3" />

<p>The softmax will output pedesttrian, car, motorcycle or background.</p>

<p>To localize the car in the image, you can change you network to have a few more output units that output a bounding box. In particular, you can have the neural network output four more numbers, and I call them bx, by, by and bw. These four numbers parameterize the bounding box of the detected object.</p>

<img width="2360" height="1574" alt="QQ_1777297661355" src="https://github.com/user-attachments/assets/cf8ed318-3c54-4b9d-9abb-135d502e8f80" />

<p>To specifying the bounding box, requires specifying the midpoint (bx, by), the height will be bh, and the width bw.</p>

<p>So training set contains not just the object class label, but it also contains four numbers. Then <b>you can use supervised learning to make your algorithm outputs not just a class label, but also the four parameters.</b></p>

<img width="638" height="562" alt="QQ_1777298162827" src="https://github.com/user-attachments/assets/bdb34614-8797-4756-a3b4-96d88c21961a" />

<p>We assume that there's at most one of these objects appears in the picture.</p>

<img width="378" height="880" alt="QQ_1777301740799" src="https://github.com/user-attachments/assets/2b120803-f741-4248-88a4-ca077b35e28f" />

<p>Finally, let's describe the loss function: </p>

<p>If y_1 = 1</p>

$$
L(\hat{y}, y) = (\hat{y}_1 - y_1)^2 + (\hat{y}_2 - y_2)^2 + \cdots + (\hat{y}_8 - y_8)^2
$$

<p>If y_1 = 0</p>

$$
L(\hat{y}, y) = (\hat{y}_1 - y_1)^2
$$

</br>

# Landmark detection

</br>

<p>You can have a neural network just output x and y coordinates of important points in image, sometimes called landmarks that you want the neural network to recognize. </p>

<p>Let's say you're building a face recognition application, and for some reason you want the algorithm to tell you where's the corner of the person's eye. You can let the final layer output two more numbers called lx and ly. If you want more corners of eye, just outputs l1x, l1y, l2x, l2y, ...</p>

<p>We assume that there're sixty four landmarks on face. And maybe some points define the jawline. <b>By selecting the number of landmarks, and generating a label training set that contains all of these landmarks, then have the neural network outputs where are all the key landmarks on the face.</b></p>

<img width="1640" height="658" alt="QQ_1777344878298" src="https://github.com/user-attachments/assets/4ce77144-1a90-454f-975c-666cc5262101" />

</br>

# Object dection

</br>

<p>Let's use a ConvNet to perform object detection, using a algorithm called the <b>sliding windows detection</b>.</p>

<img width="658" height="1104" alt="QQ_1777347173020" src="https://github.com/user-attachments/assets/f2df8b6d-2825-443c-9f67-82a47f91172e" />

<p>We have this training set. For our purpose, we use closely cropped image, meaning that the image is pretty much only the car, cut out anything else there's not part of a car. Then you can train a ConvNet.</p>

<p><b>Once you've trained up this ConvNet, you can then use it in sliding windows detection</b>. </p>

<p>Here is a test image:</p>

<img width="698" height="696" alt="QQ_1777347472282" src="https://github.com/user-attachments/assets/237c56c6-a52f-48da-a772-623ce295af1b" />

<p>What you do is picking a certain windows size. And you input into this ConvNet a small rectangular region. <b>And have the ConvNet make a prediction, is there a car?</b>.</p>

<img width="800" height="1100" alt="QQ_1777347713102" src="https://github.com/user-attachments/assets/edecd738-9751-4718-aaa9-c77d8b5e7b8f" />

<p>Then the algorithm will process input a second image, and run the ConvNet again. Until you've slid the window across every position in the image.</p>

<img width="708" height="696" alt="QQ_1777347781505" src="https://github.com/user-attachments/assets/07402136-bf60-4528-93bf-018f5779e1f9" />

<p>Repeated it, but now we use a larger window.</p>

<img width="710" height="706" alt="QQ_1777347921806" src="https://github.com/user-attachments/assets/96607228-e100-4c80-864a-4c2e24c4a826" />

<p>This algorithm has a large computational cost, but it runs okey.</p>

</br>

# Convolutional Implementation of Sliding Windows

</br>

<img width="2864" height="570" alt="QQ_1777430284630" src="https://github.com/user-attachments/assets/97406298-d9fa-4644-9c57-2cc230bbdcb9" />

<p>Let's turn these FC layers into Convolutional layers.</p>

<img width="1820" height="492" alt="QQ_1777430372956" src="https://github.com/user-attachments/assets/743f67e4-c127-4c92-bf02-15c3aacee13a" />

<p>For the next layer, let's use four hundrad 5 by 5 filters to convolve. The output dimension is going to be 1 by 1 by 400. <b>Mathematically, this is same as a fully connected layer.</b></p>

<p>To implement next conv layer, we're going to implement a 1 by 1 convolution. And then finally, <b>we're going to have another 1 by 1 filter to obtain a softmax layer.</b></p>

<img width="2900" height="522" alt="QQ_1777431173777" src="https://github.com/user-attachments/assets/3fd1a2c4-474b-4205-9ff9-4250691c683b" />

<hr>

<p>Armed of this conversion, let's see how to implement sliding windows algorithm by convolutional implementation.</p>

<img width="2684" height="458" alt="QQ_1777432090392" src="https://github.com/user-attachments/assets/95d0abdf-9647-48bb-888d-7dbbd965e8e2" />

<p>Sliding windows algorithm does a lot of <b>repeated calculation.</b></p>

<img width="424" height="518" alt="QQ_1777432473164" src="https://github.com/user-attachments/assets/47c5bd45-5231-4916-95f3-362898e3f48b" />

<p>We run sliding windows on this pretty small image, and <b>run this ConvNets four times to get four labels.</b> But a lot of computation is highly duplicated. </p>

<img width="2762" height="522" alt="QQ_1777432219653" src="https://github.com/user-attachments/assets/0637cdde-2118-48e2-a619-605ef6eaca6a" />

<p>We can <b>run the ConvNets with the same parameters.</b> Instead of forcing you to run forward propagation on four subsets independently, we combine all four into one.</p>

</br>

# Bounding box predictions

</br>

<p>Convolutional implementation of sliding windows has a problem of not quite outputting the most accurate bounding box.</p>

<img width="1172" height="1120" alt="QQ_1777433711789" src="https://github.com/user-attachments/assets/95be1d63-7991-419d-b1fc-7fbe1333db98" />

<p>In this case, none of these boxs really match up perfectly with the position of the car.</p>

<img width="764" height="710" alt="QQ_1777433824378" src="https://github.com/user-attachments/assets/d6d69ecc-6c3f-4a76-8383-c23fc9fc63a8" />

<p>And also the perfect bounding box isn't even a square.</p>

<p>One way to get this output more accurate bounding box is YOLO (You only look once).</p>

<img width="800" height="752" alt="QQ_1777447686434" src="https://github.com/user-attachments/assets/f5c9a2a5-bf7c-4d4e-a19f-43a888ca2017" />

<p>Place down a grid on this image. I'm going to use 3 by 3 grid, although in actual implementation you use a finer one (maybe 19 by 19 grid). The basic idea is you take the image classification and localization algorithm, and apply it to each of the nine grids.</p>

<p>Labels for training: (For each grid cell)</p>

`y = [pc, bx, by, bh, bw, c1, c2, c3]`

<p>The total volume of the output is going to be 3 by 3 by 8.</p>

<img width="1442" height="258" alt="QQ_1777468169201" src="https://github.com/user-attachments/assets/bf491d5c-7d34-4d6b-8c9a-a86aba8eee60" />

<p>The algorithm of this algorithm is that the neural network outputs precise bounding boxes.</p>

<p>The way you assign an object to grid cell is you look at the mid point of an object, and assign this object to whichever one grid cell contains the mid point of the object. <b>Even if the object expanse multiple grid cells.</b></p>

<img width="862" height="850" alt="QQ_1777468479505" src="https://github.com/user-attachments/assets/a278d6b3-f90b-4a6c-b89f-70ea703b97db" />

<img width="1972" height="760" alt="QQ_1777471455567" src="https://github.com/user-attachments/assets/38f9e1cc-15a8-4e7c-9ec3-dbd4bc497124" />

<p>It outputs the bounding box coordinates explicitly, this allows the neural networks to output bounding boxes with any aspect ratio as well as outputs more precise coordinates. And it isn't dictated by the stride size of your sliding windows.</p>

<p>Second, this is a convolutional implementation. You didn't run this algorithm nine times. It runs very fast, <b>so this works even for real-time object detection.</b></p>

<p></p>






























































































































































































































































































