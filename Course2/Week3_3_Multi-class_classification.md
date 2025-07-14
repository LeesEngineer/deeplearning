# SoftMax regression

</br>

<p>如果有多种可能的分类目标，有一种更普遍的逻辑回归方法叫做 softmax 回归</p>

<p>如果要识别四种类别的物品</p>

`C = #class = 4`

<img width="1552" height="310" alt="QQ_1752476955215" src="https://github.com/user-attachments/assets/7569b3aa-3235-4874-a03c-02f58a7e02f1" />

<p>在这种情况下，需要构建一个输出层是 C 的神经网络，n[L] = C 。想得到的输出层的单元告诉我们每个类别的概率 P(object | x)，输出的概率加和应该是 1，标准化的做法是使用 softmax 层</p>

```
z[L] = W[L] a[L - 1] + b[L] // (4, 1) vector
a[L] = g(z[l])
"""
Activation function:
    t = e^{(z^{[L]})}
    a^{[L]} = \frac{t_i}{\sum^4_{i = 1}{t_i}}
"""
```

<p>计算幂，再进行归一化，可以把该过程总结为一个 softmax 激活函数</p>

<img width="600" height="478" alt="QQ_1752477490329" src="https://github.com/user-attachments/assets/e67719a9-aa73-417d-8c66-2432751bd09b" />

<p>上图为一个没有隐藏层的 softmax 回归，这就是一种泛化的逻辑回归，使用一种类似线性的决策边界</p>

<p>如果有更多隐藏层和隐藏单元，就可以用更复杂的非线性决策平面来区分</p>

</br>

# Training a softmax classifier

</br>

$$
a^{[L]} = g^{[L]}(z^{[L]}) = 
\begin{bmatrix}
    0.8 \\
    0.1 \\
    0.05 \\
    0.05 
\end{bmatrix}
$$

<p>softmax 对应 hardmax，hardmax 将矢量 z 映射到矢量 [1 0 0 0]，将最大元素对应位置置为 1，其余置为 0。softmax 得到的 z 中的概率值相对平和</p>

<p>softmax 回归是 logistic 回归从二分类到多分类的推广</p>

<p>讨论如何训练一个包含 softmax 输出层的神经网络</p>

</br>

## Loss Function

</br>

$$
y =
\begin{bmatrix}
  0 \\  
  1 \\  
  0 \\ 
  0
\end{bmatrix}
$$

$$
a^{[L]} = \hat y  =
\begin{bmatrix}
  0.8 \\  
  0.1 \\  
  0.05 \\ 
  0.05
\end{bmatrix}
$$

$$
L(\hat y, y) = - \sum^4_{j = 1} y_j log \hat y_j
$$

<p>通过例子来理解：y1 = y3 = y4 = 0，只有 y2 = 1。所以和式中只剩下 `- y_2 log \hat y_2 = - log \hat y_2`</p>

<p>意味着如果想要通过梯度下降来使损失减小，只能通过增大 `\hat y_2`</p>

<p>更一般的说法是，损失函数的功能是查看训练集的真实分类值，并令其对应的概率值尽可能的大</p>

`J = 1/m \sum L(yhat, y)`

</br>

## Backprop

</br>

```
dZ[L] = \yhat - y // (4, 1) vector
```













































































































































































































































































































































































































