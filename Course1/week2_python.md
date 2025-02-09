# 矢量化 Vectorization

</br>

<p>矢量化就是一项让你的代码变得更高效的艺术。在深度学习的实际应用中，可能会遇到大量的训练数据，因为深度学习算法在这个情况下表现更好，所以代码的运行速度非常重要</p>

![QianJianTec1738987067310](https://github.com/user-attachments/assets/5c0fe87a-6078-44b8-95d9-46eff20ecc41)

![QianJianTec1738987060280](https://github.com/user-attachments/assets/aceae586-9cf2-4bc4-a014-1f03d9fffea6)

```
Non-vectorized
z = 0
for i in range(n_x):
    z += w[i] * x[i]
z += b
```

```
Vectorized: much faster
z = np.dot(w, x) + b
//w^T x
```

<p>可能听过这样的话，大规模的深度学习使用了 GPU 和图像处理单元实现，但是在 Jupyter notebook 上面实现的话，这里只有 CPU。CPU 和 GPU 都有并行化的指令，有时叫做 SIMD 指令，这代表了一个单独指令多维数组。基础意义在于如果使用了 built-in 函数，像 np.dot 或者并不要求你实现循环的函数，将使得 python 中的 numpy 充分利用并行化去更快计算，在 GPU 和 CPU 上面计算，GPU 被标记更快擅长 SIMD 计算但是 CPU 事实上也不是太差</p>

<p><b>whenever possible,avoid explicit for-loops</b></p>

</br>

# Logistic regression derivation

</br>

<p>如下是计算逻辑回归中的导数部分的代码</p>

```
J = 0, db = 0
dw1 = 0, dw2 = 0
for i = 1 to m:
    z_i = w^T * x_i + b
    a_i = sigmoid(z_i)
    J += -[ y_i log(a_i) + (1 - y_i) log(1 - a_i) ]
    dz_i = a_i - y_i
    db += dz_i
    dw1 += x1_i * dz_i, dw2 += x2_i * dz_i //second for-loop:n = 特征值数量
J = J / m, db = db / m
dw1 = dw1 / m, dw2 = dw2 / m
```

<p>消灭第二个 for-loop，做法是不再将 dw1，dw2 等显式初始化为零，移除然后令 dw 成为一个向量</p>

```
J = 0, db = 0
dw = np.zeros((n, 1))
for i = 1 to m:
    z_i = w^T * x_i + b
    a_i = sigmoid(z_i)
    J += -[ y_i log(a_i) + (1 - y_i) log(1 - a_i) ]
    dz_i = a_i - y_i
    db += dz_i
    dw += x_i * dz_i
J = J / m, db = db / m
dw /= m
```

<p>事实上，可以做得更好，实现一种没有 for-loop 的实现方式</p>

</br>

# Vectorizing Logistic Regression

</br>

<p>继续讨论逻辑回归的向量化实现，使得其可以被用于处理整个训练集，也就是说，可以用梯度下降的一次迭代来处理整个训练集，不需要一个 for-loop。后面讨论神经网络时，也不需要一个 for-loop</p>

<p>首先来看逻辑回归的前向传播，我们有 m 个训练样本，为了预测第一个样本，需要如下计算：</p>

![QianJianTec1739024651367](https://github.com/user-attachments/assets/884f313d-413d-46a3-8038-1603af60919f)


![QianJianTec1739024678938](https://github.com/user-attachments/assets/69e1a587-fa0e-4dc3-83f8-c73ad320d776)

<p>然后预测第二个，第三个，第 m 个，都需要做这个计算。不过为了实现前向传播，即计算出 m 个训练样本的预测结果，有不用 for-loop 的实现方法</p>

<p>首先回忆一下，曾把矩阵 X 定义为训练的输入值</p>

![QianJianTec1739025101548](https://github.com/user-attachments/assets/815e02ae-3024-468a-a549-08ead674243c)

<p>计算 z1, z2, z3，只需要用一行代码：首先构造一个 1 * m 维的行向量，方便计算 z1, z2, zm，实际上可以表达成</p>

![QianJianTec1739025403330](https://github.com/user-attachments/assets/43aaa3e2-c5b6-404f-aa06-dc8822036045)

`Z = np.dot(w.T, X) + b`

<p>关于求 a：就像把所有 z 排列在一起可以得到 Z，排列所有 a，得到 A， 在作业中，将看到如何实现一个输入输出为向量值的 sigmoid 函数，所以将 Z 作为 sigmoid 的输入值，会非常高效的得到 A</p>

<p>下一步，证明还可以用向量化来高效计算反向传播计算导数</p>

</br>

# Vectorizing Logistic Regression's Gradient Computation

</br>

<p>你已经看到了如何通过向量化计算预测，同时计算出整个训练集的激活值 a，现在讨论如何用向量化计算全部 m 个训练样本的梯度（同时计算）</p>

<p>在讲梯度下降时，dz_1 = a_1 - y_1，以此类推。所以我们可以定义一个新矩阵（m 维行向量）：</p>

`dZ = [dz_1, dz_2, ... , dz_m]`

<p>同理定义 A，Y，基于这样的定义，得到 dZ = A - Y，仅需一行代码，可以完成所有计算</p>

<p>在之前的实现中，已经去掉了一个 for-loop，但仍有一个遍历训练集的循环：</p>

<p>通过如下的操作把他们向量化，对于计算 db 的向量化实现，只需要对所有 dz 求和再除以 m。</p>

`db = np.sum(dZ) / m`

`dw = (X * dz^T) / m`

<p>sum up:</p>

```
Z = np.dot(w.T, X) + b
A = sigmoid(Z)
dz = A - Y
dw = (X * dZ.T) / m
db = np.sum(dZ) / m
w := w - alpha * dw
b := b - alpha * db
//完成了前向传播和反向传播
```

<p>实现了逻辑回归梯度下降的一次迭代。虽然说过要尽量避免 for-loop，但要实现梯度下降的多次迭代，那么仍然需要使用 for-loop，去迭代指定的次数</p>

</br>

# 关于 Python/Numpy 向量说明

</br>

```
import numpy as np
a = np.random.randn(5)
print(a.shape) --> (5,)
```

<p>a 的形状是这种(5, )的结构，这在 python 中叫做秩为 1 的数组，它既不是行向量，也不是列向量，这会导致一些不直观的影响</p>

```
print(a.T) //结果与 print(a) 一样，所以 a 和 a 的转置看起来一样

print(np.dot(a, a.T)) //你也许认为 a 乘以 a 转置，或者说 a 的外积，是一个矩阵，但得到的却是一个数字
```

<p>所以在编写神经网络时，不要使用这种数据结构，即形如(n,)这样秩为 1 的数组，应如此：</p>

`a = np.random.randn(5, 1)` 
`a = a.reshape((5, 1))`






















