<p>训练神经网络需要指定权重的初始值。选择合适的初始化方法将有助于神经网络学习。不同的初始化方法将导致不同的结果</p>

- 加速梯度下降的收敛

- 增加梯度下降收敛到较低训练（和泛化）误差的几率

```
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec

%matplotlib inline
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# load image dataset: blue/red dots in circles
train_X, train_Y, test_X, test_Y = load_dataset()
```

</br>

# 神经网络模型

</br>

<p>以一个三层神经网络为例子讲解</p>

<p>要尝试的初始化方法有：</p>

- Zeros initialization: 在输入参数中设置 initialization = "zeros"

- Random initialization: 在输入参数中设置 initialization = "random"，<b>这会将权重初始化为较大的随机值。</b>

- He initialization: 在输入参数中设置 initialization = "he"，This initializes the weights to random values scaled according to a paper by He

```
def model(X, Y, learning_rate = 0.01, num_iterations = 15000, print_cost = True, initialization = "he"):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (containing 0 for red dots; 1 for blue dots), of shape (1, number of examples)
    learning_rate -- learning rate for gradient descent 
    num_iterations -- number of iterations to run gradient descent
    print_cost -- if True, print the cost every 1000 iterations
    initialization -- flag to choose which initialization to use ("zeros","random" or "he")
    
    Returns:
    parameters -- parameters learnt by the model
    """
        
    grads = {}
    costs = [] # to keep track of the loss
    m = X.shape[1] # number of examples
    layers_dims = [X.shape[0], 10, 5, 1]
    
    # Initialize parameters dictionary.
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        a3, cache = forward_propagation(X, parameters)
        
        # Loss
        cost = compute_loss(a3, Y)

        # Backward propagation.
        grads = backward_propagation(X, Y, cache)
        
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the loss every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)
            
    # plot the loss
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters
```

</br>

# Zero initialization

</br>

<p>要初始化两类参数，权重矩阵和偏差向量</p>

```
# GRADED FUNCTION: initialize_parameters_zeros 

def initialize_parameters_zeros(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    parameters = {}
    L = len(layers_dims)            # number of layers in the network
    
    for i in range(1, L):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(i)] = np.zeros((layers_dims[i], layers_dims[i - 1]))
        parameters['b' + str(i)] = np.zeros((layers_dims[i], 1))
        ### END CODE HERE ###
    return parameters
```

<p>接下来看到效果并不理想，</p>

```
parameters = model(train_X, train_Y, initialization = "zeros")
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)
```

![1a4ff0bfa7ffbf30d7d57de91b42676e](https://github.com/user-attachments/assets/e099848d-45e4-4a61-90d2-c63bd229a059)

<p>成本并没有真正降低</p>

<p>Lets look at the details of the predictions and the decision boundary</p>

```
print ("predictions_train = " + str(predictions_train))
print ("predictions_test = " + str(predictions_test))
```

```
Output:
predictions_train = [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0]]
predictions_test = [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
```

![QQ_1746155556367](https://github.com/user-attachments/assets/2293429a-e46f-4975-90ba-be0b10df36c1)

<p>The model is predicting 0 for every example.</p>

<p>一般来说，将所有权重初始化为零会导致网络无法打破对称性。这意味着每一层中的每个神经元都会学到相同的东西，你实际上是在训练一个每一层都只有一个神经元（n[l] = 1）的神经网络，而这个网络的能力也不会比一个线性分类器（例如逻辑回归）更强。</p>

- 权重 W^{[l]} 应该随机初始化，以打破对称性。

- 但是偏置 b^{[l]} 初始化为零是可以的。只要 W^{[l]} 是随机初始化的，就能打破对称性。

</br>

## 关于成本函数并没有下降

</br>

<p>关于成本没有下降，有人给我提出是因为梯度消失导致的</p>

<p>来看看反例</p>

```
X = np.array([[1, -1], [1, -1]])  # shape = (2, 2)
Y = np.array([[1, 0]])            # shape = (1, 2)

W1 = np.zeros((3, 2))  # 3 hidden units, input size 2
b1 = np.zeros((3, 1))
W2 = np.zeros((1, 3))
b2 = np.zeros((1, 1))
```

### 前向传播

<p>第一层：</p>

$$
Z1 = W1 \cdot X + b1 = 0 \Rightarrow A1 = \sigma(Z1) = 0.5
$$

```
A1 = np.array([[0.5, 0.5],
               [0.5, 0.5],
               [0.5, 0.5]])  # shape: (3, 2)
```

<p>第二层：</p>

$$
Z2 = W2 \cdot A1 + b2 = 0 \Rightarrow A2 = \sigma(Z2) = 0.5
$$

```
A2 = np.array([[0.5, 0.5]])  # shape: (1, 2)
```

### 反向传播

<p>第 2 层（输出层）</p>

$$
dZ2 = A2 - Y = \begin{bmatrix} 0.5 - 1 & 0.5 - 0 \end{bmatrix} = \begin{bmatrix} -0.5 & 0.5 \end{bmatrix}
$$

<p>计算</p>

$$
dW2 = \frac{1}{m} \cdot dZ2 \cdot A1^T
$$

$$
dW2 = \frac{1}{2} \cdot \begin{bmatrix} 0 & 0 & 0 \end{bmatrix}
= \begin{bmatrix} 0 & 0 & 0 \end{bmatrix}
$$

<p>第 1 层（隐藏层）</p>

$$
dA1 = W2^T \cdot dZ2 = 0 \cdot dZ2 = 0
\Rightarrow dZ1 = dA1 \cdot \sigma{\prime}(Z1)
\Rightarrow dZ1 = 0
\Rightarrow dW1 = 0
$$

<p>dW[l] 都为 0</p>

<p>我保留意见吧，所以你的意思是对称性导致学不到不同的东西，也就是相当于只有一个神经元在学习，你的意思是还是能学到东西的，但我们前面看到了梯度都消失了，根本学不到东西啊。所以到底是学不到东西还是学到的东西有限，我前面自己写的案例是，成本函数没有任何下降，给我的答案就是学不到任何东西</p>

<p>你给我的答案是：</p>

- 不是“对称性本身”导致梯度为 0
- 而是“对称性 + 某些输入 + 某些激活函数”综合起来，导致梯度为 0

<p>我保留意见吧</p>

</br>

# Random initialization

</br>

<p>为了打破对称性，我们随机初始化权重。你将看到如果权重被随机初始化，但值非常大，会发生什么情况。</p>

<p>将权重初始化为较大的随机值（按 10 倍缩放），并将偏差初始化为零。用于np.random.randn(..,..) * 10权重，np.zeros((.., ..))用于偏差。</p>

```
# GRADED FUNCTION: initialize_parameters_random

def initialize_parameters_random(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    np.random.seed(3)               # This seed makes sure your "random" numbers will be the as ours
    parameters = {}
    L = len(layers_dims)            # integer representing the number of layers
    
    for i in range(1, L):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(i)] = np.random.randn(layers_dims[i], layers_dims[i - 1]) * 10
        parameters['b' + str(i)] = np.zeros((layers_dims[i], 1))
        ### END CODE HERE ###

    return parameters
```

```
parameters = model(train_X, train_Y, initialization = "random")
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)
```

```
Cost after iteration 0: inf
Cost after iteration 1000: 0.6240946882166043
Cost after iteration 2000: 0.5978449711241829
Cost after iteration 3000: 0.5636338302631672
Cost after iteration 4000: 0.550097327262185
Cost after iteration 5000: 0.5443542140109684
Cost after iteration 6000: 0.5373689058507384
Cost after iteration 7000: 0.4703439954192491
Cost after iteration 8000: 0.39768133854217425
Cost after iteration 9000: 0.39344534852511953
Cost after iteration 10000: 0.39201679908257836
Cost after iteration 11000: 0.38915469271083064
Cost after iteration 12000: 0.38612747395619024
Cost after iteration 13000: 0.3849707683892173
Cost after iteration 14000: 0.3827517632656006
```

![QQ_1746172175080](https://github.com/user-attachments/assets/38ddaae0-5dde-4c6b-a9cd-3217baa773dd)

<p>打破了对称性，给出了更好的结果</p>

![QQ_1746172238232](https://github.com/user-attachments/assets/e253e0c6-000e-4a53-b7e5-dd342468a018)

- 一开始的成本就很高。这是因为，当权重随机值较大时，最后一个激活函数 (Sigmoid) 输出的结果在某些样本中非常接近 0 或 1，而当它判断错误时，就会给该样本带来非常高的损失。事实上，当 log(a[3]) = log(0) 时，损失趋于无穷大

- 初始化不良会导致梯度消失/爆炸，这也会减慢优化算法的速度。

- <b>如果对该网络进行更长时间的训练，您将看到更好的结果，但使用过大的随机数进行初始化会减慢优化速度。</b>

- 将权重初始化为非常大的随机值效果不佳。希望用较小的随机值初始化效果更好。重要的问题是：这些随机值应该多小？来看看 He initialization

</br>

# He initialization

</br>

<p>Xavier 初始化和 He 初始化类似，只不过 Xavier 初始化对权重 W^{[l]} 使用的缩放因子是 \sqrt{1/\text{layers\_dims}[l-1]}，而 He 初始化使用的是 \sqrt{2/\text{layers\_dims}[l-1]}。</p>

$$
\sqrt{1/\text{layersdims}[l-1]}
$$

$$
\sqrt{2/\text{layersdims}[l-1]}
$$

<p>This function is similar to the previous initialize_parameters_random(...). The only difference is that instead of multiplying np.random.randn(..,..) by 10, you will multiply it by 
\sqrt{2/\text{layers\_dims}[l-1]}，which is what He initialization recommends for <b>layers with a ReLU activation.</b></p>

```
# GRADED FUNCTION: initialize_parameters_he

def initialize_parameters_he(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1 # integer representing the number of layers
     
    for i in range(1, L + 1):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(i)] = np.random.randn(layers_dims[i],layers_dims[i - 1]) * np.sqrt(2.0 / layers_dims[i - 1])
        parameters['b' + str(i)] = np.zeros((layers_dims[i], 1))
        ### END CODE HERE ###
        
    return parameters
```

```
parameters = model(train_X, train_Y, initialization = "he")
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

Output:







parameters = model(train_X, train_Y, initialization = "he")
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)
第 0 次迭代后的成本：0.8830537463419761
迭代 1000 次后的成本：0.6879825919728063
迭代 2000 次后的成本：0.6751286264523371
迭代 3000 次后的成本：0.6526117768893807
迭代 4000 次后的成本：0.6082958970572938
迭代 5000 次后的成本：0.5304944491717495
迭代 6000 次后的成本：0.4138645817071794
迭代 7000 次后的成本：0.3117803464844441
迭代 8000 次后的成本：0.2369621533032255
迭代 9000 次后的成本：0.18597287209206836
迭代 10000 次后的成本：0.1501555628037181
迭代 11000 次后的成本：0.12325079292273544
迭代 12000 次后的成本：0.09917746546525934
迭代 13000 次后的成本：0.08457055954024277
迭代 14000 次后的成本：0.07357895962677369
```

![QQ_1746173149577](https://github.com/user-attachments/assets/831ec029-a95b-4f7f-bdad-e50eb1a6b59c)

![QQ_1746173166890](https://github.com/user-attachments/assets/18234452-b996-4da1-9ecd-5d6445b9b4b7)

<p>The model with He initialization separates the blue and the red dots very well in a small number of iterations.</p>

</br>

# Summary

```
3-layer NN with zeros initialization:
    On the train set:
    Accuracy: 0.5
    On the test set:
    Accuracy: 0.5
fails to break symmetry

3-layer NN with large random initialization:
    On the train set:
    Accuracy: 0.83
    On the test set:
    Accuracy: 0.86
too large weights 

3-layer NN with He initialization:
    On the train set:
    Accuracy: 0.9933333333333333
    On the test set:
    Accuracy: 0.96
recommended method
```

- Different initializations lead to different results

- Random initialization is used to break symmetry and make sure different hidden units can learn different things

- Don't intialize to values that are too large

- He initialization works well for networks with <b>ReLU activations</b>

























































































































































































































































































































































































































































































































