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






























































































































































































































































































































































































































































































































