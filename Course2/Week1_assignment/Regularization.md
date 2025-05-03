</br>

# Regularization

</br>

<p>深度学习模型具有非常强的灵活性和容量，如果训练数据集不够大，过拟合就可能成为一个严重的问题。确实，它在训练集上的表现很好，但学习到的网络无法很好地泛化到那些从未见过的新样本上！</p>

```
# import packages
import numpy as np
import matplotlib.pyplot as plt
from reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
from reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters
import sklearn
import sklearn.datasets
import scipy.io
from testCases import *

%matplotlib inline
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
```

<p>Problem Statement: You have just been hired as an AI expert by the French Football Corporation. They would like you to recommend positions where France's goal keeper should kick the ball so that the French team's players can then hit it with their head.They give you the following 2D dataset from France's past 10 games.</p>

![QQ_1746174519904](https://github.com/user-attachments/assets/93ba9ad1-05aa-4e62-adc4-ded209434205)

![QQ_1746174776358](https://github.com/user-attachments/assets/8e10a9b0-4c1c-4ddf-9a96-7e9a9323440e)

<p>这个数据集有点嘈杂，但看起来将左上半部分（蓝色）与右下半部分（红色）分开的对角线效果很好</p>

</br>

# Non-regularized model

</br>

<p>以实现一个网络，可用于：</p>

- in regularization mode -- by setting the lambd input to a non-zero value.

- in dropout mode -- by setting the keep_prob to a value less than one

<p>先尝试不带任何正则化的模型。然后实现：</p>

- L2 正则化——函数：“ compute_cost_with_regularization()”和“ backward_propagation_with_regularization()”

- Dropout——函数：“ forward_propagation_with_dropout()”和“ backward_propagation_with_dropout()”

```
def model(X, Y, learning_rate = 0.3, num_iterations = 30000, print_cost = True, lambd = 0, keep_prob = 1):

    grads = {}
    costs = []                            # to keep track of the cost
    m = X.shape[1]                        # number of examples
    layers_dims = [X.shape[0], 20, 3, 1]
    
    # Initialize parameters dictionary.
    parameters = initialize_parameters(layers_dims)

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        if keep_prob == 1:
            a3, cache = forward_propagation(X, parameters)
        elif keep_prob < 1:
            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)
        
        # Cost function
        if lambd == 0:
            cost = compute_cost(a3, Y)
        else:
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)
            
        # Backward propagation.
        assert(lambd==0 or keep_prob==1)    # it is possible to use both L2 regularization and dropout, 
                                            # but this assignment will only explore one at a time
        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(X, Y, cache)
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)
        
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the loss every 10000 iterations
        if print_cost and i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)
    
    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters
```

```
parameters = model(train_X, train_Y)
print ("On the training set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

Output:
Cost after iteration 0: 0.6557412523481002
Cost after iteration 10000: 0.1632998752572417
Cost after iteration 20000: 0.13851642423261248

On the training set:
Accuracy: 0.9478672985781991
On the test set:
Accuracy: 0.915
```

![QQ_1746175290621](https://github.com/user-attachments/assets/a2f29349-e9bf-4a50-945c-510c20d79bff)

<p>The train accuracy is 94.8% while the test accuracy is 91.5%. This is the baseline model (you will observe the impact of regularization on this model).</p>

<p>Baseline Model：基线模型指的是在做改进或比较之前使用的最基础的模型</p>

- 最简单的结构（比如没有正则化、没有调参）

- 最基本的训练方式

- 作为后续模型改进的参照对象

![QQ_1746175465663](https://github.com/user-attachments/assets/0a2dc930-63a5-4134-b626-e872d6a557db)

<p>The non-regularized model is obviously overfitting the training set. It is fitting the noisy points! Lets now look at two techniques to reduce overfitting.</p>

</br>

# L2 Regularization

</br>

<p>L2 正则化适当地修改成本函数：</p>

![QQ_1746175646382](https://github.com/user-attachments/assets/38c58558-5ad0-4c0f-82f0-6d06b54555dc)

<p>use 'np.sum(np.square(Wl))' to calculate L2 regularization cost</p>

```
def compute_cost_with_regularization(A3, Y, parameters, lambd):

    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    
    cross_entropy_cost = compute_cost(A3, Y) # This gives you the cross-entropy part of the cost
    
    ### START CODE HERE ### (approx. 1 line)
    #L2_regularization_cost = None
    L2_regularization_cost = (lambd / (2 * m)) * (
        np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))
    )
    ### END CODER HERE ###
    
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost
```

<p>因为你改变了成本，所以你也必须改变反向传播，所有梯度都必须根据这个新的成本进行计算。这些更改仅涉及 dW1、dW2 和 dW3。对于每个项，你必须添加正则化项的梯度</p>

$$
\frac{d}{dW}(\frac{1}{2}\frac{\lambda}{m}W^2) = \frac{\lambda}{m}W
$$

```
def backward_propagation_with_regularization(X, Y, cache, lambd):

    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    
    ### START CODE HERE ### (approx. 1 line)
    #dW3 = 1./m * np.dot(dZ3, A2.T) + None
    dW3 = 1./m * np.dot(dZ3, A2.T) + (lambd / m) * W3
    ### END CODE HERE ###
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    ### START CODE HERE ### (approx. 1 line)
    #dW2 = 1./m * np.dot(dZ2, A1.T) + None
    dW2 = 1./m * np.dot(dZ2, A1.T) + (lambd / m) * W2
    ### END CODE HERE ###
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    ### START CODE HERE ### (approx. 1 line)
    #dW1 = 1./m * np.dot(dZ1, X.T) + None
    dW1 = 1./m * np.dot(dZ1, X.T)  + (lambd / m) * W1
    ### END CODE HERE ###
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients
```

<p>Let's now run the model with L2 regularization(\lambda = 0.7). The model() function will call:</p>

- compute_cost_with_regularization instead of compute_cost

- backward_propagation_with_regularization instead of backward_propagation

```
parameters = model(train_X, train_Y, lambd = 0.7)
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

Output:
第 0 次迭代后的成本：0.6974484493131264
第 10000 次迭代后的成本：0.2684918873282238
第 20000 次迭代后的成本：0.26809163371273004

在训练集上：
准确率：0.9383886255924171
在测试集上：
准确率：0.93
```

![QQ_1746177080659](https://github.com/user-attachments/assets/8956fe99-152c-45dd-a7a9-646ce97a6dae)

<p>You are not overfitting the training data anymore</p>

![QQ_1746177133732](https://github.com/user-attachments/assets/3bd448df-33f7-4ea1-9e6e-8ada6d172e0e)

- The value of lambda is a hyperparameter that you can tune using a dev set.

- L2 regularization makes your decision boundary smoother. If lambda is too large, it is also possible to "oversmooth", resulting in a model with high bias.

<p>L2 正则化基于这样一个假设：权重较小的模型比权重较大的模型更简单。因此，通过在代价函数中惩罚权重的平方值，你会促使所有权重变得更小。当权重变大时，代价函数的值会大幅上升，代价变得太高！这样可以得到一个更加平滑的模型——当输入发生变化时，输出的变化也会更缓慢。</p>

<p>L2 正则化的影响包括：</p>

- 对代价计算的影响：在代价函数中会加入一个正则化项。

- 对反向传播的影响：计算权重矩阵梯度时会多出一些额外的项。

- 对权重的影响（称为“权重衰减”）：权重会被压缩到更小的值。

</br>

# Dropout

</br>

<p>dropout is a widely used regularization technique that is specific to deep learning. It randomly shuts down some neurons in each iteration.</p>

<p>At each iteration, you shut down (= set to zero) each neuron of a layer with probability 1-keep_prob or keep it with probability keep_prob (50% here). 丢弃的神经元对迭代的前向传播和后向传播中的训练均无贡献。</p>

<p>当你关闭一些神经元时，你实际上是在修改你的模型。dropout 背后的原理是，在每次迭代中，你训练一个只使用部分神经元子集的不同模型。使用 dropout，你的神经元对另一个特定神经元的激活变得不那么敏感，因为那个神经元可能随时被关闭。</p>

</br>

## Forward propagation with dropout

</br>

<p>实现带dropout的前向传播。你使用一个三层神经网络，并将dropout应用于第一和第二个隐藏层。我们不会在输入层和输出层应用dropout。</p>

```
def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
    
    np.random.seed(1)
    
    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    ### START CODE HERE ### (approx. 4 lines)         # Steps 1-4 below correspond to the Steps 1-4 described above. 
    D1 = np.random.rand(A1.shape[0], A1.shape[1])              # Step 1
    D1 = (D1 < keep_prob).astype(int)                          # Step 2
    A1 = A1 * D1                                               # Step 3
    A1 = A1 / keep_prob                                        # Step 4
    ### END CODE HERE ###
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    ### START CODE HERE ### (approx. 4 lines)
    D2 = np.random.rand(A2.shape[0], A2.shape[1])              # Step 1
    D2 = (D2 < keep_prob).astype(int)                          # Step 2
    A2 = A2 * D2                                               # Step 3
    A2 = A2 / keep_prob                                        # Step 4
    ### END CODE HERE ###
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    
    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
    
    return A3, cache
```

</br>

## Backward propagation with dropout

</br>

<p>Add dropout to the first and second hidden layers, using the masks D[1] and D[2] stored in the cache.</p>

- You had previously shut down some neurons during forward propagation, by applying a mask D[1] to A1. In backpropagation, you will have to shut down the same neurons, by reapplying the same mask D[1] to dA1.

- During forward propagation, you had divided A1 by keep_prob. In backpropagation, you'll therefore have to divide dA1 by keep_prob again.

```
def backward_propagation_with_dropout(X, Y, cache, keep_prob):

    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    dA2 = np.dot(W3.T, dZ3)
    ### START CODE HERE ### (≈ 2 lines of code)

    dA2 = dA2 * D2                       # Step 1: apply the dropout mask
    dA2 = dA2 / keep_prob               # Step 2: scale the values
    ### END CODE HERE ###
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
    ### START CODE HERE ### (≈ 2 lines of code)

    dA1 = dA1 * D1                       # Step 1: apply the dropout mask
    dA1 = dA1 / keep_prob               # Step 2: scale the values
    ### END CODE HERE ###
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients
```

<hr>

<p>Let's now run the model with dropout (keep_prob = 0.86). It means at every iteration you shut down each neurons of layer 1 and 2 with 24% probability. The function model() will now call:</p>

- forward_propagation_with_dropout instead of forward_propagation.

- backward_propagation_with_dropout instead of backward_propagation.

```
parameters = model(train_X, train_Y, keep_prob = 0.86, learning_rate = 0.3)
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

Output:
Cost after iteration 0: 0.6543912405149825
Cost after iteration 10000: 0.0610169865749056
Cost after iteration 20000: 0.060582435798513114

On the train set:
Accuracy: 0.9289099526066351
On the test set:
Accuracy: 0.95
```

![QQ_1746178649916](https://github.com/user-attachments/assets/7b3ba703-58a2-42aa-8b89-f8892e7f34d3)

<p>Dropout works great!Your model is not overfitting the training set and does a great job on the test set.</p>

![QQ_1746178693454](https://github.com/user-attachments/assets/570a07d8-358c-4b74-b0a0-3db9dbe46f23)

</br>

## Dropout Summary

</br>

- A common mistake when using dropout is to use it both in training and testing. You should use dropout (randomly eliminate nodes) only in training.

- Deep learning frameworks like tensorflow, PaddlePaddle, keras or caffe come with a dropout layer implementation. Don't stress - you will soon learn some of these frameworks.


<p>Note:</p>

- 只在训练时使用 Dropout，测试时不要使用（即不要随机丢弃节点）。

- Dropout 需要在前向传播和反向传播时都进行处理。

- 在训练时，每一层的输出要除以 keep_prob，以保持激活值的期望不变。

</br>

# Summary

</br>

<p>Regularization will drive your weights to lower values to help you reduce overfitting</p>

```
3-layer NN without regularization:
    95%
    91.5%
3-layer NN with L2-regularization:
    94%
    93%
3-layer NN with dropout:
    93%
    95%
```

<p>Note that regularization hurts training set performance! This is because it limits the ability of the network to overfit to the training set. But since it ultimately gives better test accuracy, it is helping your system.</p>




























































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































