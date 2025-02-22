# 前言

<br>

<p>构建第一个神经网络，有一个隐藏层。将看到此模型与使用逻辑回归实现的模型之间存在很大差异。</p>

<p>内容：</p>

- 实现具有单个隐藏层的二类分类神经网络

- 使用具有非线性激活函数的单元，例如 tanh

- 计算交叉熵损失

- 实现前向和后向传播

</br>

# package

</br>

```
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
# 自己的包
from planar_utils import plot_decision_boundary, sigmoid, load_extra_datasets
from testCases_v2 import *
```

<p>sklearn提供了简单、高效的数据挖掘和数据分析工具。</p>

```
# 加上
def load_planar_dataset():
    np.random.seed(1)
    m = 400
    N = int(m / 2)
    D = 2
    X = np.zeros((m, D))
    Y = np.zeros((m, 1), dtype = "uint8")
    a = 4
    
    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j
        
    X = X.T
    Y = Y.T
    
    return X,Y

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
```

</br>

# Dataset

</br>

<p>获取您将要处理的数据集</p>

`X, Y = load_planar_dataset()`

<p>可视化数据：</p>

`plt.scatter(X[0, :], X[1, :], c=Y[0, :], s=40, cmap=plt.cm.Spectral);`

![2bbd684f-25fe-43c5-b9e4-35201335f9b8](https://github.com/user-attachments/assets/5f00f77f-dc0e-429c-80c4-79c9fb0fee5a)

</br>

# Simple Logistic Regression

</br>

<p>在构建完整的神经网络之前，先看看逻辑回归如何解决这个问题。使用 sklearn 的内置函数来做到这一点。运行下面的代码来在数据集上训练逻辑回归分类器。</p>

```
clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T, Y.T[:, 0]);
```

<p>打印 accuracy</p>

```
LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")

# Output: Accuracy of logistic regression: 47 % (percentage of correctly labelled datapoints)
```

<p>绘制模型的决策边界</p>

```
plot_decision_boundary(lambda x: clf.predict(x), X, Y[0, :])
plt.title("Logistic Regression")
```

![604da860-0824-4bd8-b257-abab27f038b6](https://github.com/user-attachments/assets/d79e41c1-46f1-4fa3-8dd2-071cb46458bd)

<p><b>我们的数据集不是线性可分的，因此逻辑回归表现不佳。</b></p>

</br>

# Neural Network model¶

</br>

<p>逻辑回归在“花卉数据集”上效果不佳。故训练一个具有单个隐藏层的神经网络。</p>

![b993e954-4477-4e3e-ac12-06b907458756](https://github.com/user-attachments/assets/293baca2-1939-4271-baa7-b4cd8b3ed43a)

<p>数学上讲：</p>

![f0ea41a9-e28c-47bb-9fbf-23061e9f3892](https://github.com/user-attachments/assets/22264e30-47c4-4489-b812-ead23142c740)

<p>得到所有示例的预测，计算成本</p>

![94154653-08a7-48c4-a7f9-bf1ac67b1a11](https://github.com/user-attachments/assets/68a152d7-8550-4140-b4ff-39ecdb6ae26a)

<p>构建神经网络的方法：</p>

1. 定义神经网络结构（输入单元数、隐藏单元数等）

2. 初始化模型参数

3. 循环：
    - 实现前向传播
    - 计算损失
    - 实现后向传播以获取梯度
    - 更新参数（梯度下降）

<p>构建完各个部分后，合并到函数 nn_model 中，学习了正确的参数后就可以对新数据进行预测</p>

</br>

## Defining the neural network structure

</br>

<p>定义三个变量：</p>

- n_x：输入层的大小

- n_h：隐藏层的大小（设置为 4）

- n_y：输出层的大小

```
def layer_sizes(X, Y):

    n_x = X.shape[0] # size of input layer
    n_h = 4
    n_y = Y.shape[0] # size of output layer

    return (n_x, n_h, n_y)
```

</br>

## Initialize the model's parameters

</br>

<p>使用随机值初始化权重矩阵：np.random.randn(a,b) * 0.01 随机初始化形状为 (a,b) 的矩阵。</p>

<p>把偏差向量初始化为零：np.zeros((a,b))用零初始化形状为 (a,b) 的矩阵。</p>

```
def initialize_parameters(n_x, n_h, n_y):
    
    np.random.seed(2) # we set up a seed so that your output matches ours although the initialization is random.

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters
```

</br>

## The Loop

</br>

</br>

### Implement forward propagation

</br>

<p>反向传播中所需的值存储在“cache”中。cache将作为反向传播函数的输入。</p>

```
def forward_propagation(X, parameters):

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache
```

</br>

### Compute cost J

</br>

<p>已经得到了 A[2]，可以计算成本函数</p>

<p>实现交叉熵损失的方法有很多种，给出一种</p>

```
logprobs = np.multiply(np.log(A2),Y)
cost = - np.sum(logprobs)
# no need to use a for loop!
```

```
def compute_cost(A2, Y, parameters):
    
    m = Y.shape[1] # number of example

    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
    cost = -np.sum(logprobs) / m
    
    cost = np.squeeze(cost)

    assert(isinstance(cost, float))
    
    return cost
```

</br>

### implement backward propagation

</br>

![deb3c0a2-5c0b-462e-a3bd-660546c8efa6](https://github.com/user-attachments/assets/013e87b2-1dca-4eb4-a3c4-f786c19e8047)

```
def backward_propagation(parameters, cache, X, Y):

    m = X.shape[1]
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis = 1, keepdims = True) / m
    dZ1 = np.multiply(np.dot(W2.T,dZ2), 1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis = 1, keepdims = True) / m
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads
```

</br>

### Implement the update rule

</br>

<p>如下：具有良好学习率（收敛），不良学习率（发散）。</p>

![63c71132-8dbf-4a97-a674-d88e4936ac82](https://github.com/user-attachments/assets/41328358-a3cb-400b-b0dc-6ad4018b04d2)

```
def update_parameters(parameters, grads, learning_rate = 1.2):

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * grads["dW1"]
    b1 = b1 - learning_rate * grads["db1"]
    W2 = W2 - learning_rate * grads["dW2"]
    b2 = b2 - learning_rate * grads["db2"]
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters
```

</br>

### Integrate

</br>

```
def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):

    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):

        A2, cache = forward_propagation(X, parameters)

        cost = compute_cost(A2, Y, parameters)

        grads = backward_propagation(parameters, cache, X, Y)

        parameters = update_parameters(parameters, grads)

        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters
```

</br>

### Predict

</br>

<p>通过构建 predict() 使用你的模型进行预测。使用前向传播来预测结果</p>

```
def predict(parameters, X):

    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)

    return predictions
```

</br>

### Summary

</br>

```
# Build a model with a n_h-dimensional hidden layer
parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)
predictions = predict(parameters, X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')
```

</br>

# Tuning hidden layer size

</br>

<p>观察模型在不同隐藏层大小下的不同行为。</p>

```
plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations = 5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y[0, :])
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
```

```
1 个隐藏单元的准确度：67.5 % 
2 个隐藏单元的准确度：67.25 % 
3 个隐藏单元的准确度：90.75 % 
4 个隐藏单元的准确度：90.5 % 
5 个隐藏单元的准确度：91.25 % 
20 个隐藏单元的准确度：90.0 % 
50 个隐藏单元的准确度：90.75 %
```

![170f4c35-6be7-4300-be54-0e82948a49a0](https://github.com/user-attachments/assets/216a93e9-3e13-40f0-aa86-0446fb6be93d)

![cb381505-7ac6-4ad3-ae50-fc3f455244e7](https://github.com/user-attachments/assets/733e094d-ef80-4f5d-83d3-461849e0cac8)

![16466428-705f-4159-9917-499e3c8b5c6d](https://github.com/user-attachments/assets/f5ca7eb6-3d06-406f-ba79-f60bbcb8793d)

![bc73d1d9-61d7-4143-9487-56e484114b62](https://github.com/user-attachments/assets/f2656c0a-cf8c-4d6f-89bb-355330fea206)

<p>解释：</p>

- 较大的模型（具有更多隐藏单元）能够更好地适合训练集，直到最终最大的模型过度拟合数据。

- 最佳隐藏层大小似乎在 n_h = 5 左右。事实上，这里的值似乎能很好地拟合数据，而不会产生明显的过度拟合。

- 稍后还将了解正则化，它可以让您使用非常大的模型（例如 n_h = 50）而不会出现过度拟合。

<p>如今已经学会：</p>

- 构建具有隐藏层的完整神经网络

- 充分利用非线性单元

- 实现前向传播和反向传播，并训练神经网络

- 查看改变隐藏层大小的影响，包括过度拟合。


















































































































