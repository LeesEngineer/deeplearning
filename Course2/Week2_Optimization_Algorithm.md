<p>本周要学习加快神经网络训练素的的优化算法，在大数据领域中，深度学习表现的并不算完美，能够训练集于大量数据的神经网络，而用大量数据就会很慢</p>
 
</br>

# Mini-batch gradient descent

</br>

<p>Vectorization allows you to efficiently compute on m examples，但是如果 m 非常大(5000000?)，速度依旧很慢。</p>

<p>对整个训练集运用梯度下降法，<b>必须先处理整个训练集，才能在梯度下降中往前走一步</b>，所以算法实际上是可以加快的，让梯度下降在处理完整个巨型的训练集之前就开始生效</p>

<p>首先将训练集拆分为更小的训练集，即小批量训练集(mini-batch)，比如说每一个微型训练集只有 1000 个训练样例(x1 to x1000, x1001 to x2000)，</p>

<p>新记号：X{1} 代表 x1 to x1000，类推。y 也做相应的拆分处理</p>

`Mini-batch t: X^{t}, Y^{t}`

<p>Batch: 同时处理整个训练集，名字由来就是同时处理整个训练集批次</p>

<p>mini-batch: 每次只处理一个小批量样例 X{t}, Y{t}，而不是一次处理完整个训练集 X Y  </p>

</br>

## How it works

</br>

<p>在训练集上运行小批量梯度下降法的时候，每个子集都要运行一遍</p>

<p>for 循环里面要做的基本上就是用 (X{t}, Y{t}) 做一次梯度下降，用向量化的方法同时处理 1000 个样例</p>

```
for t = 1, ..., 5000
    ForwardProp on X{t}
        Z[1] = W[1] X{t} + b[1]
        A[1] = g[1](Z[1])
        ...
        A[L] = g[L](Z[L])
        //上面这些都是矢量化方法（1000 examples）

    Compute cost
    Backprop to compute gradients
    W[l] := W[l] - alpha dW[l], b[l] := b[l] - alpha db[l]
```

$$
Cost: J^{\{t\}} = \sum_{i = 1}^{l = 1000}L(\hat{y}^{(i)}, y^{(i)}) / 1000 + \frac{\lambda}{2000} \sum_l \|W^{[l]}\|_F^2
$$

<p>这是小批量梯度下降算法处理训练集一轮的过程，也叫做训练集的一次遍历（epoch），遍历是指过一遍训练集。在批量梯度下降法中对训练集的一轮处理只能得到一步梯度逼近，而小批量梯度下降法中对训练集的一轮 epoch，可以得到 5000 步梯度逼近</p>

<p>当你有一个大型训练集时，小批量梯度下降法比梯度下降法快得多</p>

</br>

# Understanding mini-batch gradient descent

</br>

<p>在批量梯度下降算法中，每一次迭代将遍历整个训练集，用 J 来表示代价函数，那么他应该随着迭代单调递减。<b>如果某一次迭代他的值增加了，那么一定是哪里错了，比如学习率太大</b></p>

![QQ_1746254521244](https://github.com/user-attachments/assets/3b91b052-81c7-4496-8d2f-d72c9ea1e31f)

<p>而在小批量梯度下降中，同样画图就会发现并不是每一次迭代代价函数的值都会减小。从细节来看，每次迭代都是对 X{t} Y{t} 的处理，对通过他们计算出来的代价函数 J{t} 进行画图，就好像每次迭代都使用不同的训练集（也就是使用不同的 mini-batch），就会看到这样的图，他的趋势是向下的，但是也会有很多噪声</p>

![QQ_1746255297935](https://github.com/user-attachments/assets/00391dab-ed0a-4bb5-aeb0-ffda7636bf40)

<p>如果使用小批量梯度下降算法，经过几轮训练后，对 J{t} 作图很可能就像这样，并不是每次迭代都会下降，但是整体趋势必须是向下的</p>

<p>而他之所以有噪声，可能和计算代价函数时所用的那个批次 X{t}, Y{t} 有关，让代价函数的值或大或小。也有可能这个批次含有一些标签错误的数据，导致代价函数有一点高</p>

</br>

## Choosing your mini-batch size

</br>

<p>必须定义的一个参数是 mini-batch 的大小</p>

<p>一种极端情况 If mini-batch size = m，其实就是批量梯度下降。(X{1}, Y{1}) = (X, Y)</p>

<p>另一种极端情况则是把 mini-batch 设置为 1，会得到一种叫随机梯度下降的算法，每一条数据都是一个 mini-batch</p>

<p>看看两种方法在优化代价函数时有什么不同</p>

<p>批量梯度下降算法（假设从边缘开始）噪声相对小些，每一步相对较大，并且最终可以达到最小值</p>

![QQ_1746427972940](https://github.com/user-attachments/assets/c967f7d3-8b52-490a-9b45-710bb9f7e76f)

<hr>

<p>相对的，随机梯度下降算法（假设从这里开始），每一次迭代就在一个样本上做梯度下降，大多时候可以达到全局最小。但有时也可能因为某组数据不太好，把你指向一个错误的方向，因此随机梯度算法的噪声会非常大。一般来说会沿着一个正确的方向，而且<b>随机梯度下降算法最后也不会收敛到一个点，一般会在最低点附近摆动</b></p>

![QQ_1746430603323](https://github.com/user-attachments/assets/f3152a33-ebdc-49dd-b372-19064e143300)

<p><b>如果使用随机梯度下降算法，使用一个样本来更新梯度，这没有问题，而且可以通过选择比较小的学习率来减少噪声。但随机梯度下降有一个很大的缺点是失去了利用向量加速运算的机会</b></p>

<p>实际上 mini-batch 的大小会在这两个极端之间，既可以使用向量加速，也可以不用等整个训练集遍历完一遍才运行梯度下降</p>

![QQ_1746430664125](https://github.com/user-attachments/assets/8e580ae9-4be5-4e3b-a66b-40549eae8990)

<p>不能保证总能达到最小值，但相比随机梯度下降，他的噪声会更小，而且不会总在最小值附近摆动（如果有什么问题，可以缓慢的减小学习率，将会介绍学习率衰减）</p>

<p>准则：</p>

- 如果你的训练集较小（2000？），就使用批量梯度下降

- 较大的训练集，一般选择 64 - 512 作为 mini-batch 的大小（这是因为计算机内存的布局和访问模式，把 mini-batch 的大小设置为 2 的幂数会更快），1024 还是比较罕见的

- 确保所有的 X{t} Y{t} 是可以放进 CPU/GPU 内存的

</br>

# Exponentially weighted averages（指数加权平均值）（指数加权滑动平均）

</br>

<p>展示几个优化算法，比梯度下降更快，为了理解他们，需要用到一种叫做指数加权平均的操作，在统计学上也被称为指数加权滑动平均，将使用这个概念来构建更复杂的优化算法</p>

![QQ_1746431767667](https://github.com/user-attachments/assets/09eeca13-385e-40fb-9c00-70ca286e7b2b)

<p>这些数据可能有噪声，如果想计算数据的趋势（即局部平均或滑动平均）</p>

```
V_0 = 0
V_1 = 0.9 V_0 + 0.1 theta_1
V_2 = 0.9 V_0 + 0.1 theta_2
...
V_t = 0.9 V_{t - 1} + 0.1 theta_t
```

<p>如果这样计算，并把结果用红线描述出来，就会得到一个滑动平均的图，称为每日温度的指数加权平均</p>

![QQ_1746861086109](https://github.com/user-attachments/assets/fcdc0d98-82bc-40ce-92fe-7b083e7fee07)

$$
V_t = \beta V_{t - 1} + (1 - \beta)\theta_t
$$

<p>计算这个公式的时候，可以认为 V_t 近似于 1 / (1 - beta) 天温度的平均</p>

<p>beta 等于 0.98 时，相当于计算了 50 天的平均值，（<b>因为 0.98 的五十次方才能使该天的值足够小，之后他们将更快的衰减</b>）画出来：</p>

![QQ_1746861417375](https://github.com/user-attachments/assets/dd826008-65f9-4694-a794-ea5de67f38c3)

<p>注意：</p>
 
- 当 beta 的值很大的时候，得到的曲线会更平滑，因为对更多天数的温度做了平均处理，因此曲线就波动更小

- 另一方面，曲线会右移，因为在一个更大的窗口内计算平均温度。通过在更大的窗口内计算平均，这个指数加权平均的公式，在温度变化时，适应地更加缓慢，会造成一些延迟。原因是当 beta = 0.98 时，之前的值具有更大的权重，而当前值的权重就很小

<p>取另一个极端，beta = 0.5，变成只对两天平均，仅仅两天的平均，就是在很小的窗口内计算平均，得到的结果中会有很多噪声，更容易收到异常值的影响，但可以更快的适应温度变化</p>

![QQ_1746862195945](https://github.com/user-attachments/assets/6d984f99-a89d-4d0b-93f5-ee0fd8d54634)

<hr>

<p>通过调整这个参数（也就是会在后面学习算法中看到的一个超参数）可以得到略微不同的收益（红色曲线最好，对温度的平均最好）</p>

<p>关于使用：高效，因为几乎只占一行代码，只需要基于数值不断重写运算即可。如果计算一个滑动窗口：算出十天五十天的平均值，通常会给出一个更好的估计，但缺点是保存所有值，需要更多内存，更难实现，花费更高</p>

```
V = 0
loop
{
    get next theta_t
    V := beta V + (1 - beta) theta_t
}
```

<p>接下来学习偏差修正，结合两者能够实现一个更好的优化算法</p>

</br>

# Bias correction in exponentially weighted average

</br>

<p>偏差校正可以让这些平均值的计算更加准确</p>

<p>前面讲过，红色是 0.9，绿色是 0.98。实际上，如果实现该公式 `V_t = beta V_{t - 1} + (1 - beta) theta_t`，如果 beta 等于 0.98，不会得到绿色曲线，而是会得到紫色曲线</p>

![QQ_1746868441869](https://github.com/user-attachments/assets/74b1611a-8fd4-4396-aeb8-c70c52a5dde1)

<p>曲线的起点很低。</p>

<p>在实现移动平均线时，V0 = 0，所以 V1 实际等于 0.02*theta1，V2 = 0.0196*theta1 + 0.02*theta2。V2 将远低于theta1和theta2，所以 V2 对前两天并不是一个很好的估计</p>

<p>有一种方法可以修改这个估计值，使其变得更好，<b>尤其是在估算的初始阶段</b></p>

<p>与其取 Vt，不如用 Vt 除以 (1 - beta^t)</p>

```
t = 2:
1 - beta = 1 - (0.98)^2 = 0.0396
V2 / (0.0396)
```

<p>当 t 变大，beta^t 将接近 0，这就是为什么 t 变大时，紫线几乎与绿线重合。<b>但是在学习的初始阶段，当你还在预热估计值时，偏差校正可以更好的估计（从紫线变成绿线）</b></p>

</br>

# Gradient descent with momentum

</br>

<p>动量梯度下降法几乎总会比标准的梯度下降法快，算法的主要思想是，计算梯度的指数加权平均，然后使用该梯度更新权重</p>

<p>假设从一点开始执行梯度下降，经过梯度下降的一次迭代后，无论是批量还是小批量下降，结果都可能朝向一个达到椭圆的另一边的方向，在进行一步梯度下降，可能到了这里，以此类推。会发现梯度下降计算很多步，向着最小值缓慢的震荡前进。<b>这种上下的震荡会减慢梯度下降的速度，同时也让你无法使用较大的学习率（为了避免真当过大）</b></p>

![QQ_1746880243020](https://github.com/user-attachments/assets/6217c16c-5087-4642-8675-fa9990baed57)

<p>下面是实现栋梁梯度下降的步骤；</p>

```
Momentum:
On iteration t:
    //initialize V_dW with 0
    Compute dW, db on current (mini-)batch
    V_dW = beta V_dW + (1 - beta) dW //计算 dW 的滑动平均
    V_db = beta V_db + (1 - beta) db

    W := W - alpha * V_dW
    b := b - alpha * V_db
```

<p>这样做可以让梯度下降的每一步变得平滑</p>

![QQ_1746880280700](https://github.com/user-attachments/assets/408cc327-a55b-4d63-acc1-4ac87950bbbd)

<p>在数次迭代之后，发现栋梁梯度下降的每一步，在垂直方向上的震荡非常小，且在水平方向上的运动得更快。让算法选择更直接的路径</p>

<p>关于偏差校正，实际上，当在实现动量梯度下降时，没讲过有人会做偏差校正（但 V_dW 还是要初始化为 0），因为在十轮迭代后，滑动平均已经就绪，不再是一个偏差估计</p>

</br>

# RMSprop

</br>

<p>已经学习了如何使用动量来加速梯度下降，还有一种 RMSprop(全称为均方根传递--Root Mean Square prop)也可以加速梯度下降</p>

![QQ_1749816128323](https://github.com/user-attachments/assets/35f94267-b763-4742-8883-84c9a863ddda)

<p>假设坐标为 b->y w->x，希望减慢 b 方向上的学习，同时加速或至少不减慢 w 方向的学习。这就是 RMSprop 要做的</p>

```
On iteration t:
    Compute dW, db on current mini-batch
    S_dW = beta_2 S_dW + (1 - beta_2) dW ** 2
    S_db = beta_2 S_db + (1 - beta_2) db ** 2
    // 逐元素的平方操作

    W := W - alpha dW / sqrt(S_dW + epsilon)
    b := b - alpha db / sqrt(S_db + epsilon)
    // epsilon 是为了防止除以的值过小，是一个非常小的值（1e-8）
```

<p>希望在 W 方向上学习速率较快，在 b 方向上减少震荡，工作原理：S_dW 会相对较小，S_db 会相对较大（根据导数），结果是垂直方向上的更新量会除以一个较大的数，有助于减少震荡，水平方向的更新量会除以一个较小的数</p>

<p>直观理解是：在容易出现震荡的维度里，会得到一个更大的 S 值（即导数平方的加权平均），最后抑制了这些出现震荡的方向</p>

![QQ_1749818091000](https://github.com/user-attachments/assets/ac76cd3c-2502-44c9-bc3a-c48ddfdcb813)

<p>另一个收益是可以使用更大的学习率，而不用担心在垂直方向上发散</p>

</br>

# Adam optimization algorithm

</br>

<p>Adam 基本上是利用动量和 RMSprop 组合在一起</p>

```
V_dW = 0, S_dW = 0, V_db = 0, S_db = 0
On iteration t:
    Compute dW, db using current mini-batch
    V_dW = beta_1 V_dW + (1 - beta_1) dW, V_db = beta_1 V_db + (1 - beta_1) db
    S_dW = beta_2 S_dW + (1 - beta_2) dW^2, S_db = beta_2 S_db + (1 - beta_2) db^2
    // 在 Adam 中实现了偏差校正
    Vc_dW = V_dW / (1 - beta_1^t), Vc_db = V_db / (1 - beta_1^t)
    Sc_dW = S_dW / (1 - beta_2^t), Sc_db = S_db / (1 - beta_2^t)

    W := W - alpha * Vc_dW / sqrt(Sc_dW + epsilon), b := b - alpha * Vc_db / sqrt(Sc_db + epsilon)
```

<p>该学习算法已被证明对各种架构的许多不同神经网络非常有效，有许多超参数</p>

```
alpha : needs to be tune
beta_1 : 0.9
beta_2 : 0.999
epsilon : 1e-8
```

<p>beta_1 计算导数的平均值。beta_2 用于计算平方的指数加权平均值</p>

</br>

# Learning rate decay

</br>

<p>学习率衰减可能有助于加快学习算法速度，如果要慢慢的降低 Alpha，那么在最初阶段，当你的 Alpha 仍然很高时，依旧可以进行相对较快的学习。但是随着 Alpha 变小，走的步数会越来越慢，最终会在最小值附近更窄的区域内摆动，而不是徘徊的很远</p>

![QQ_1750064252191](https://github.com/user-attachments/assets/dd420426-e76a-42b4-bfe5-6a2e1c679b77)

<p>慢慢降低 Alpha 背后的理念是：在最初的学习阶段，有能力迈出更大的步伐，但是随着学习接近趋同，学习速度变慢就可以迈出更小的步伐</p>

$$
\alpha = \frac{1}{1 + decayrate * epochnum}\alpha_0
$$

<p>除了这种公式，还有别的方法。如，指数衰减：</p>

$$
\alpha = 0.95^{epochnum} * \alpha_0
$$

<p>这将迅速降低学习率</p>

<p>learning rate decay 的优先级较低</p> 

</br>

# The problem of local optima

</br>

<p>在深度学习的早期阶段，人们常常担心优化算法会陷入糟糕的局部最优之中</p>




























































































































































