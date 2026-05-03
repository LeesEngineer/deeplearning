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

</br>

# Intersection Over Union (IoU) Function

</br> 

<p>We use a function called intersection over union both for evaluating object detection algorithm, and adding another component to your algorithm.</p>

<p>What the IoU function does is computing the intersection over union of these two bounding boxes.</p>

<img width="614" height="618" alt="QQ_1777515451047" src="https://github.com/user-attachments/assets/fa9f387c-eb14-4ed8-b6f5-9fc65a7a95f1" />

<p>Correct if IoU >= 0.5. 0.5 is human-chosen convention. The higher the IoUs, the accurate the bounding box.</p>

<p>This is one way to map localization to accuracy, where you just count up the times of algorithm correctly detects and localizes an object.</p>

</br>

# Non-max Suppression

</br>

<p>Your algorithm may find multiple detection of the same object. Non-max suppression makes sure that your algorithm detects each object only once.</p>

<p>Technically, only one of the grid cells should predict that there is a car. But in practice, you're running a object classification and localization algorithm <b>for every one of these grid cells. Another grid cells might think that the center of the car is in it, and so might the other cells.</b></p>

<img width="1104" height="876" alt="QQ_1777517506121" src="https://github.com/user-attachments/assets/90bed26d-1f83-4a73-b09e-40016f0e47d5" />

<img width="1094" height="874" alt="QQ_1777518224548" src="https://github.com/user-attachments/assets/b3da8149-8f3f-413d-8966-a20411a0a704" />

<p>When you run the algorithm, you might end up with multiple detections of each object. So what non-max suppression does is cleaning up these detection. It first looks at the probablities associated with each of these detections count on the p_c. It first takes the largest one, and says that's my most confident detection. Then non-max suppression part looks at all of the remaining rectangles. <b>All the ones that have a high IoU with our most confident detection will get suppressed. And get rid of them.</b></p>

<img width="1090" height="858" alt="QQ_1777519546271" src="https://github.com/user-attachments/assets/dbdf211e-5c7c-43d9-83d8-1dd2cdf355d9" />

<hr>

<p>First run the algorithm on this 19 by 19 grid cells image. You get 19 by 19 by 8 output.</p>

<img width="674" height="672" alt="QQ_1777519680350" src="https://github.com/user-attachments/assets/b6e64a08-6d1d-4d1b-93a5-b215e008c64a" />

<p>Then discard all boxes with p_c <= 0.6</p>

<p>While there are any remaining boxes:</p>

- Pick the box with the largest p_c, and output that as a prediction.

- Discard any remaining box with IoU >= 0.5 with the box output.

</br>

# Anchor Boxes

</br>

<p>Each grid cells can only detect only one object. What if a grid cell want to detect multiple objects.</p>

<img width="716" height="718" alt="QQ_1777529330160" src="https://github.com/user-attachments/assets/1b5dd74d-8bce-40d1-a355-f905df9c1716" />

<p>Notice that the midpoint of the pedestrian and the midpoint of the car fall into the same grid cell.</p>

<p>The idea of anchor boxes is predefining two different shapes called anchor box.</p>

<img width="1396" height="730" alt="QQ_1777529530549" src="https://github.com/user-attachments/assets/3b8d06dc-3a63-4bc6-bc38-d55298490429" />

<p>Now associate two predictions with the two anchor boxes. Define y as followed:</p>

<img width="914" height="738" alt="QQ_1777529643461" src="https://github.com/user-attachments/assets/41c58c74-4232-4e53-b019-d093c97dd6f6" />

<p>Because the shape of pedestrian is more similar to the anchor box 1. So use the first eight number to encode the pedestrian. And the next eight numbers are all associated with the detected car. The output is going to be 3 by 3 by 16.</p>

<p>Previously, each object in training image is assigned to grid cell that contains that object's midpoint. With the two anchor boxes, each onject in training image is assigned to grid cell that contains object's midpoint and anchor box for the grid cell with highest IoU. Each object assigned to a (grid cell, anchor box) pair. That's how that object gets encoded in the target label.</p>

<p>Anchor box allows your learning algorithm to specialize better in particular if your data set has some tall skinny objects like pedestrians and some wide objects cars .</p>

<p>People used to choose anchor box by hand.</p>

</br>

# YOLO Algorithm

</br>

<p>Let's see how you construct your training set.</p>

<img width="1958" height="1100" alt="QQ_1777723504473" src="https://github.com/user-attachments/assets/cffa2eff-09d1-48d7-b3a4-fb2cf5b6f0d8" />

<p>For each grid cell, get 2 predicted bounding boxes.</p>

<img width="608" height="604" alt="QQ_1777729742694" src="https://github.com/user-attachments/assets/7b23d468-4798-4173-95e5-78f2ede9a869" />

<p>Get rid of low probability predictions.</p>

<img width="606" height="592" alt="QQ_1777729850818" src="https://github.com/user-attachments/assets/bc3e5e45-4fe4-4070-8079-6bbc5f542ebc" />

<p>For each class (pedestrian, car, motorcycle) use non-max suppression to generate final predictions.</p>

<img width="606" height="604" alt="QQ_1777729956738" src="https://github.com/user-attachments/assets/cb4f15ba-5906-4327-81a4-b33a87397518" />

</br>

# Region proposals

</br>

<p>I tend to use the region proposal set of algorithms a bit less often.</p>

<p>If you use the sliding windows and run it across all of these different windows to see if there is anything. <b>One downside is that it classifies a lot of regions where there's clearly no object.</b></p>

<img width="698" height="610" alt="QQ_1777731719857" src="https://github.com/user-attachments/assets/6fd41500-7c43-45dd-bbe2-d753a78f95dd" />

<p>So there's an algorithm called R-CNN. It tries to pick a few regions that makes sense to run your ConvNet classifier, rather that running your sliding windows on every single window.</p>

<p>The way to perform the region proposals is to run an algorithm called segmentation algorithm.</p>

<img width="672" height="586" alt="QQ_1777804202818" src="https://github.com/user-attachments/assets/f690eaed-459d-4378-bb40-5a6585abc1c7" />

<p>You get this image. To figure out where could be objects.</p>




































































































































































































































