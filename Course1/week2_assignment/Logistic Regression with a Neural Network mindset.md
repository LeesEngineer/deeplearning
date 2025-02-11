# Logistic Regression with a Neural Network mindset

</br>

<p>构建学习算法的总体架构，包括：</p>

- 初始化参数

- 计算成本函数及其梯度

- 使用优化算法（梯度下降）

<p>将上述所有函数集中到一个主模型函数中</p>

</br>

# Package

</br>

```
# 1.
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
```

- numpy是 Python 进行科学计算的基础包。

- h5py是一个与存储在 H5 文件中的数据集交互的常用包。

- matplotlib是一个著名的 Python 图形绘制库。

- 这里使用PIL和scipy在最后用你自己的图片来测试你的模型。

</br>

# 问题集概述

</br>

<p>构建一个简单的图像识别算法，可以正确地将图片分类为猫或非猫。通过运行以下代码加载数据：</p>

```
# 2.
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
```

<p>在图像数据集的末尾添加了“_orig”，因为我们要对它们进行预处理。预处理后，我们将得到 train_set_x 和 test_set_x（标签 train_set_y 和 test_set_y 不需要任何预处理）。</p>

<b>深度学习中的许多软件错误都源于矩阵/向量维度不匹配。如果您能保持矩阵/向量维度一致，那么就可以大大消除许多错误。</b>

```
# 3.
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
print(train_set_x_orig.shape)    # OUTPUT: (209, 64, 64, 3)
```

<p>现在应该在形状为 (num_px, num_px, 3) 的 numpy 数组中重塑形状为 (num_px * num_px * 3, 1)。此后数据集就是一个 numpy 数组，每一列代表一个扁平图像，应该有 m 列</p>

<p>重塑训练和测试数据集，A trick when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b * c * d, a) is to use:</p>

`X_flatten = X.reshape(X.shape[0], -1).T`

```
# 4.
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T
```

<p>为了表示彩色图像，必须为每个像素指定红、绿、蓝通道 (RGB)，因此像素值实际上是一个由 0 到 255 之间的三个数字组成的向量。</p>
  
<p>机器学习中一个常见的预处理步骤是将数据集居中并标准化，这意味着您从每个示例中减去整个 numpy 数组的平均值，然后将每个示例除以整个 numpy 数组的标准差。但对于图片数据集，将数据集的每一行除以 255（像素通道的最大值）更简单、更方便，并且效果几乎一样好。</p>

```
# 5.
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.
```

<p>预处理新数据集的常见步骤是：</p>

- 找出问题的维度和形状（m_train、m_test、num_px……）

- 重塑数据集，使得每个示例现在都是一个大小为 (num_px * num_px * 3, 1) 的向量

- “标准化”数据

</br>

# 学习算法的总体架构

</br>

<p>将使用神经网络思维构建逻辑回归。下图解释了为什么逻辑回归实际上是一个非常简单的神经网络</p>

<img width="518" alt="LogReg_kiank" src="https://github.com/user-attachments/assets/7f36ef23-8f3e-40d1-8f82-5712958442c0" />

<p>该算法的数学表达式，举个例子（x^i）</p>

![QianJianTec1739195242635](https://github.com/user-attachments/assets/e1e9950f-07b0-443f-b0e4-9782c70b780c)

![QianJianTec1739195251659](https://github.com/user-attachments/assets/9df5e7db-25f6-41a2-8b47-6ba26d6155b0)

![QianJianTec1739195261026](https://github.com/user-attachments/assets/b29f507c-42ab-4986-9a3f-ad80a83a8199)

<p>然后通过对所有训练示例求和来计算成本：</p>

![QianJianTec1739195271397](https://github.com/user-attachments/assets/6fadad26-900e-4b16-9259-d5747f0e1dd6)

<p>关键步骤：</p>

- 初始化模型的参数

- 通过最小化成本来学习模型的参数

- 使用学习到的参数进行预测（在测试集上）

- 分析结果并得出结论

</br>

# 构建算法的各个部分

</br>

<p>建立神经网络的主要步骤是：</p>

1. 定义模型结构（例如输入特征的数量）

2. 初始化模型的参数

3. 环形：
   - 计算当前损耗（前向传播）
  
   - 计算当前梯度（反向传播）
  
   - 更新参数（梯度下降）
  
<p>通常会分别构建 1 - 3，然后集成到一个 model 函数中</p>

</br>

## 辅助函数

</br>

```
# 6.
def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    s = 1.0 / (1.0 + np.exp(-1.0 * z))
    ### END CODE HERE ###
    
    return s
```

</br>

## 初始化参数

</br>

<p>在下面的单元格中实现参数初始化。必须将 w 初始化为零向量。</p>

```
# 7.
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    w = np.zeros((dim,1))
    b = 0
    ### END CODE HERE ###

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b
```

</br>

## 前向传播和后向传播

</br>

<p>参数已初始化，您可以执行“前向”和“后向”传播步骤来学习参数。实现一个 propagate() 计算成本函数及其梯度的函数</p>

<p>前向传播：</p>

- 得到 X

- 计算 A = sigmoid(w^T X + b) = (a^1, a^2, ... , a^m)

- 计算成本函数：

![QianJianTec1739196944241](https://github.com/user-attachments/assets/9bcc6328-7edf-41a6-b153-c9c6ab07e331)

<p>以下是将使用的两个公式：</p>

![QianJianTec1739196964282](https://github.com/user-attachments/assets/95948057-3e54-4696-85aa-d111f6718993)

![QianJianTec1739196986560](https://github.com/user-attachments/assets/33680f7d-dfa4-4e93-9fc1-90778f28ba6c)

```
# 8.
def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    ### START CODE HERE ### (≈ 2 lines of code)

    A = sigmoid(np.dot(w.T, X) + b)                                    # compute activation
    cost = -(1.0 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))                                # compute cost

    ### END CODE HERE ###
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    ### START CODE HERE ### (≈ 2 lines of code)

    dw = (1.0 / m) * np.dot(X, (A - Y).T)
    db = (1.0 / m) * np.sum(A - Y)

    ### END CODE HERE ###

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost
```

</br>

## 优化-Optimization

</br>

- 以初始化参数

- 以计算成本函数及其梯度

- 现在可以使用梯度下降来更新参数

<p>写下优化函数，学习 w 和 b，最小化成本函数 J</p>

```
# 9.
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation (≈ 1-4 lines of code)
        ### START CODE HERE ### 
        grads, cost = propagate(w, b, X, Y)
        ### END CODE HERE ###
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        w = w - learning_rate * dw
        b = b - learning_rate * db
        ### END CODE HERE ###
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs
```

</br>

## predict

</br>

<p>上一个函数将输出学习到的 w 和 b。我们能够使用 w 和 b 来预测数据集 X 的标签。实现该predict()函数。计算预测有两个步骤：</p>

1. 计算 Yhat = A = sigmoid(w^T X + b)

2. 将 a 的条目转换为 0（如果激活 <= 0.5）或 1（如果激活 > 0.5），将预测存储在向量中。如果您愿意，您可以在循环中Y_prediction使用if/语句（尽管也有一种方法可以将其矢量化）。

```
def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    ### START CODE HERE ### (≈ 1 line of code)
    A = sigmoid(np.dot(w.T, X) + b)
    ### END CODE HERE ###
    
    for i in range(A.shape[1]):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        ### START CODE HERE ### (≈ 4 lines of code)
        if A[0, i] > 0.5:    
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
        ### END CODE HERE ###
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction
```

</br>

## sum up

</br>

<p>你已经实现了几个函数：</p>

- 初始化 (w,b)

- 迭代优化损失以学习参数 (w,b)

- 计算成本及其梯度

- 使用梯度下降更新参数

- 使用学习到的 (w,b) 预测给定示例集的标签

</br>

# 将所有函数合并到一个模型中

</br>

<p>将看到如何按照正确的顺序将所有构建块（前面部分实现的功能）组合在一起来构建整个模型。</p>

```
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    ### START CODE HERE ###
    
    # initialize parameters with zeros (≈ 1 line of code)
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
 
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    ### END CODE HERE ###

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d
```

<p>运行以下单元来训练你的模型</p>

`d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)`

```
Cost after iteration 0: 0.693147
Cost after iteration 100: 0.584508
Cost after iteration 200: 0.466949
Cost after iteration 300: 0.376007
Cost after iteration 400: 0.331463
Cost after iteration 500: 0.303273
Cost after iteration 600: 0.279880
Cost after iteration 700: 0.260042
Cost after iteration 800: 0.242941
Cost after iteration 900: 0.228004
Cost after iteration 1000: 0.214820
Cost after iteration 1100: 0.203078
Cost after iteration 1200: 0.192544
Cost after iteration 1300: 0.183033
Cost after iteration 1400: 0.174399
Cost after iteration 1500: 0.166521
Cost after iteration 1600: 0.159305
Cost after iteration 1700: 0.152667
Cost after iteration 1800: 0.146542
Cost after iteration 1900: 0.140872
train accuracy: 99.04306220095694 %
test accuracy: 70.0 %
```

<p>训练准确率接近 100%。这是一次很好的健全性检查：模型正在运行，并且具有足够高的容量来适应训练数据。测试错误率为 68%。考虑到我们使用的数据集很小，并且逻辑回归是线性分类器，对于这个简单的模型来说，这实际上还不错。</p>

![eba5ec59-b387-4741-8569-a3a0a72f4aea](https://github.com/user-attachments/assets/2a638c18-47ac-47e2-b939-a41b4a41251b)

<p>可以看到成本在下降。这表明参数正在被学习。但是，您会发现您可以在训练集上对模型进行更多训练。尝试增加上面单元格中的迭代次数并重新运行单元格。您可能会看到训练集准确率上升，但测试集准确率下降。这称为过度拟合。</p>

</br>

# Further analysis

</br>

<p>进一步分析图像分类模型，研究学习率的可能选择</p>

<p>调整学习率可以对算法产生很大影响</p>

<p>为了使梯度下降法发挥作用，必须明智地选择学习率，alpha 决定了我们更新参数的速度，如果学习率太大，我们可能会“超过”最佳值。同样，如果学习率太小，我们将需要太多迭代才能收敛到最佳值。这就是为什么使用经过良好调整的学习率至关重要。</p>

<p>将模型的学习曲线与几种学习率进行比较。运行下面的单元格。这大约需要 1 分钟。也可以尝试除我们初始化变量learning_rates以包含的三个值以外的其他值，看看会发生什么。</p>

```
learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
```

```
学习率为：0.01
训练准确率：99.52153110047847 %
测试准确率：68.0 % 
-------------------------------------------------------
学习率为：0.001
训练准确率：88.99521531100478 %
测试准确率：64.0 % 
-------------------------------------------------------
学习率为：0.0001
训练准确率：68.42105263157895 %
测试准确率：36.0 % 
-------------------------------------------------------
```

![e0b8c919-c5ab-4ac9-a769-6251c76da8b4](https://github.com/user-attachments/assets/f97ddaa0-abeb-4b35-a0c6-2385a859c0bc)

- 不同的学习率会产生不同的成本，从而产生不同的预测结果。

- 如果学习率过大（0.01），成本可能会上下波动。它甚至可能会发散（尽管在这个例子中，使用 0.01 最终仍会得到一个良好的成本值）。

- 成本较低并不意味着模型更好。您必须检查是否存在过度拟合。当训练准确率远高于测试准确率时，就会发生这种情况。

- 在深度学习中，我们通常建议
  - 选择能够更好地最小化成本函数的学习率。
  - 如果模型过度拟合，使用其他技术来减少过度拟合。
 
</br>

# Test with my own image

```
# 导入必要的库
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 预处理图像以适配算法
fname = "images/" + my_image
image = Image.open(fname)  # 使用 Pillow 打开图像
image = image.resize((num_px, num_px))  # 调整图像大小
image = np.array(image)  # 将图像转换为 numpy 数组

# 转换形状为模型所需格式
my_image = image.reshape((1, num_px * num_px * 3)).T

# 使用模型进行预测
my_predicted_image = predict(d["w"], d["b"], my_image)

# 显示原图像并输出预测结果
plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
```




























