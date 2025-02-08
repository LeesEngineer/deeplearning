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

























