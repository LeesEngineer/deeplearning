 # 前言

</br>

<p>当在网络中组织计算的时候，经常使用所谓的前向传播，以及反向传播，所以将会了解为什么在训练神经网络时，计算可以被正向传播组织为一次前向传播过程以及一次反向传播过程</p>

</br>

# 二元分类(Binary Classification)

</br>

<p>逻辑回归是一个二元分类算法</p>

<p>这里有一个二元分类问题的例子：有一张输入照片，希望算法输出一个 0 或 1 的 label 指明图上是不是猫。用符号 y 来表示输出标签</p>

<p>先来看图像在计算机中如何表示：为了存储一张彩色图像，计算机要存储三个独立的矩阵，为了用向量表示，把矩阵的像素值展开为一个向量 x 作为算法的输入，得到一个非常长的向量，如 64 * 64 的，得到 64*64*3 = 12288。用 n_x = 12288 来表示输入特征向量 x 的维度（n_x * 1）</p>

<p>所以在二元分类的问题中，目标是学习到这样的一个分类器：输入一副以特征向量 x 表示的图像，然后预测对应的输出 y 是 1 还是 0</p>

<p>先介绍符号：</p>

- (x, y)表示单个样本，x 为 n_x 维的特征向量，y 是标签，取值 0 或 1

- m 表示样本集中单个样本的数量

<p>为了将所有的训练样本写成更加紧凑的形式，将定义一个矩阵，用 X 表示，大小是 n_x * m</p>

<p>Y = [y_1, y_2, ... , y_m]，1 * m</p>

</br>

# 逻辑回归(Logistic Regression)

</br>

<p>使用这种学习算法会得到输出标签 y ，y 在监督学习问题中全是 0 或者 1，因此这是一种针对二分类问题的算法</p>

<p>给定的输入特征向量 x 和一副图片对应，希望识别这是否是一张猫的照片，需要一种算法能够输出一个预测值，what we call it yhat，代表对真实标签 Y 的估计：当给定输入 x 时，预测 y 为 1 的概率</p>

![QianJianTec1738723511622](https://github.com/user-attachments/assets/27032a2d-937c-4bcd-b3f7-1897def5c665)

<p><b>约定逻辑回归的参数是 w，也是一个 n_x 维的向量，另外参数 b 是一个实数</b></p>

<p>因此给定了一个输入 x 以及参数 w 和 b，生成输出 yhat，有一种方法可以试试，尽管不怎么奏效，即：</p>

![QianJianTec1738732208593](https://github.com/user-attachments/assets/13f5d64c-dbd8-45f7-b29c-3c0d4b86beaa)

<p>这是输入 x 的一个线性函数输出，如果使用<b>线形回归</b>，就是这样操作的，但是这对于二分类并不是一个好的算法，因为希望 yhat 能够在 0 和 1 之间</p>

<p>所以让逻辑回归输出 yhat 等于这个值，应该应用 sigmoid 函数的结果：</p>

![QianJianTec1738732845362](https://github.com/user-attachments/assets/25cd66f3-21ee-428c-affa-07941f4f7b67)

![Logistic-curve](https://github.com/user-attachments/assets/1196475b-295d-41ae-a899-1a323ce304dc)

<p>因此当你实现逻辑回归时，目标是尽力学习到参数 w 和 b，因此 yhat 就可以很好的估计 y 等于 1 的概率</p>

<p>当实现神经网络的时候通常将参数 w 和 b 分开看待，这里的 b 对应一个偏置量</p>

![251d6648-94e4-4940-8417-c289656ff2bd](https://github.com/user-attachments/assets/e0045c81-de27-40bd-a2ad-5e7445ac9888)

<p>在一些课程中，定义了一个额外的特征 x0，并让 x0 等于 1 ，因此 x 的维度就变成了 n_x + 1 维，之后定义就变成：</p>

![QianJianTec1738733823468](https://github.com/user-attachments/assets/138201cd-a815-422f-87a4-70102d111f5b)

<p>在这种符号约定中，有一个向量参数 theta : theta_0 ,theta_1 ,... ,theta_n。其中 theta_0 代替了 b </p>

</br>

# 损失函数 & 成本函数

</br>

## 损失函数

</br>

<p>为了训练逻辑回归模型的参数 w 和 b 的逻辑回归模型，需要定义一个成本函数</p>

<p>看看可以用什么损失函数来衡量我们的预测表现如何：</p>

<p>当你的预测是 yhat，真实标签是 y 时的损失，可能是平方误差或半平方误差</p>

![QianJianTec1738744832939](https://github.com/user-attachments/assets/03d1d54c-9ccb-4111-aa1e-d1d47a715e58)

<p>事实证明你可以这样做，但在逻辑回归中，人们通常不会这么做，因为当你开始学习参数时，<b>发现优化问题变成了非凸问题，所以你最终遇到了优化问题，即遇到多个局部 optima（最优解），可能找不到全局最优解</b></p>

<p>当真正的标签时 y 时，功能失调函数 L 被称为损失函数，需要定义这个函数来衡量我们的输出 yhat 有多好，而 squared error（平方误差）似乎是一个合理的选择，只是他在下降时效果不佳。</p>

<p>因此在逻辑回归中，我们定义了一个不同的损失函数，作用与平方误差相似，<b>但会带来一个凸的优化问题</b>，优化变得更容易</p>

![QianJianTec1738745549569](https://github.com/user-attachments/assets/6306bdb9-659d-4ddc-aa40-7496960fe8d9)

<p>以下是为什么这个损失函数有意义的直觉：如果我们使用平方误差，那么你希望平方误差尽可能小，而通过如上这种逻辑回归，丢失函数也希望塔尽可能小。为了理解，来看两种情况：</p>

- 第一种情况，假设 y 等于 1：L(yhat, y) = - log yhat，所以这表示，如果<b>希望在学习过程中尝试使损失函数变小</b>，那么 y = 1 时，yhat 应尽可能大

- 同理，如果 y 等于 0，那么你的损失函数将推送参数以使 yhat 尽可能接近 0

<p>现在有很多函数大致具有这种效果</p>

</br>

## 成本函数

</br>

<p>上面的函数是针对单个训练实例定义的，衡量你在单个训练实例的表现。现在定义一个成本函数，衡量在整个训练集上的表现</p>

<p>因此，应用于参数 w 和 b 的成本函数 J：(可以展开)</p>

![QianJianTec1738751521278](https://github.com/user-attachments/assets/4c21e502-01d0-46e5-8e20-ff7d2acd4456)

<p>损失函数仅应用于单个训练实例，成本函数就是参数的成本，所以在训练逻辑回归模型时，我们将尝试找到参数 w 和 b，这些参数可以最小化总成本函数 J</p>

<p>接下来看如何将逻辑回归视为一个非常小的神经网络</p>

</br>

# 梯度下降(Gradient Descent)

</br>

## first

</br>

<p>回顾：如何通过损失函数来界定你的模型对单一样本的训练效果。代价函数可以用来衡量参数 w 与 b 在你设计的整个模型中的作用效果</p>

<p>继续来看如何使用梯度下降模型去训练或者去学习，来调整你的训练集中的参数 w 和 b</p>

![7e62e83b-65d1-4eb6-a870-cae0c959f2b1](https://github.com/user-attachments/assets/4c8a03cc-5819-4efc-bc71-924580c2958e)

<p>现在已经有了熟悉的逻辑回归算法，代价函数 J 有参数 w 和 b，可以衡量 w 和 b 在训练集上的效果，所以要使得参数 w 和 b 的设置变得合理，就要找到使得代价函数 J(w, b)尽可能小所对应的 w 和 b</p>

<p>给出梯度下降的说明：</p>

<p>在实践中 w 可能是更高的维度</p>

![E78ACB8D8B067F7C572C008F5F424AC5](https://github.com/user-attachments/assets/b62d3ca5-bc44-4ebb-b492-1f26819fc360)

<p>目的是找到 w 和 b 使得对应的代价函数 J 值是最小的</p>

<p>可以看到代价函数 J 是一个凸函数(convex function)，与如下的函数相反，他是非凸的，并且有很多不同的局部最优</p>

![3973d8b0-34e9-4af5-bce4-8e9bf671f389](https://github.com/user-attachments/assets/6b100a3c-b9ba-4f24-b121-c8460c0815ca)

<p>因此我们的成本函数 J(w, b)，之所以定义为凸函数，一个重要的原因是我们使用对于逻辑回归这个特殊代价函数 J 导致的</p>

</br>

## second

</br>

<p>为了找到优的参数值，会用一些初始值(随意，但一般是 0)来初始化 w 和 b</p>

<p>梯度下降法会以初始点开始，然后朝<b>最陡的</b>下坡方向走一步，需要迭代这一过程</p>

<p>这里忽略 b，仅用一维曲线代替多维曲线：</p>

![f99644c7-8587-4602-bd96-7fe4fdfbbd5c](https://github.com/user-attachments/assets/b063db5a-4d51-4875-8e7d-9fe8507bf97b)

<p>梯度下降将重复执行以下更新的操作（在算法收敛到 optima 之前会重复这样做）</p>

![QianJianTec1738844070954](https://github.com/user-attachments/assets/d09b5269-7ed5-4161-9278-d5dd03eaefed)

<p>b 同理</p>

<p>alpha 表示学习率，控制在每一次梯度下降法中步长大小</p>

<p>我们希望得到在当前参数条件下，斜率是怎样的，之后我们可以采用下降素的最快的步长</p>

<p>w 和 b 的迭代公式是实际迭代更新参数时进行的操作</p>

</br>

# 计算图

</br>

## first

</br>

<p>神经网络的计算过程：由正向传播进行前向计算来计算神经网络的输出，以及反向传播计算来计算梯度和微分</p>

<p>计算图解释了为什么以这种方式来组织</p>

<p>为演示计算图，来看一个简单的逻辑回归或单层神经网络</p>

```
J(a, b, c) = 3(a + bc)

u = bc
v = a + u
J = 3v
```

![0fad82ee-e380-488b-8685-5c0e6ce54991](https://github.com/user-attachments/assets/2bbe060d-90ab-4e32-a3fb-a296d857af2b)

<p>a = 5, b = 3, c = 2 --> J = 33（一次正向传播）</p>

<p>在逻辑回归的情况下，J 是我们试图最小化的 cost 函数</p>

</br>

## second（反向传播）

</br>

<p>用例子说明如何使用计算图来计算函数 J 的导数</p>

![QianJianTec1738849277525](https://github.com/user-attachments/assets/75ae77c3-5338-4b70-8d5d-1f308b5ef32c)

<p>计算最终输出变量（而这通常是你最关心的变量）对于 v 的导数，就是一次反向传播</p>

<p>用链式法则算其他</p>

![QianJianTec1738851283597](https://github.com/user-attachments/assets/3803da5c-41c3-402c-b02c-62ae8460a0f4)

<p>引入新记号 dvar：finaloutputvar 对其他变量的导</p>

</br>

# 梯度下降逻辑回归

</br>

<p>将讨论在实现逻辑回归时，如何计算导数来实现梯度下降，重点在于梯度下降的关键方程（我承认使用计算图对于逻辑回归的梯度下降有些大材小用）</p>

<p>之前建立了如下的逻辑回归方程：</p>

![05e923f8-c674-466b-a7b3-ca766bf19799](https://github.com/user-attachments/assets/eb51ebd4-ac1b-4ea3-aa3b-95feb27c7d6e)

<p>换个例子，假如有两个特征 x1，x2：</p>

![8bdabee2-dd19-4bde-b882-fc15b6ec08e8](https://github.com/user-attachments/assets/a4c23578-dc3b-41a8-8fad-4e7895878d4d)

<p>在逻辑回归中，要做的就是修改参数 w 和 b，来减少损失函数。之前讲前向传播的例子中，讲了如何计算单个样本的损失函数，现在看看如何反向计算导数：</p>

<p>首先计算损失函数对于 a 的导数</p>

![QianJianTec1738852640297](https://github.com/user-attachments/assets/84275b77-c3f3-4402-b37b-88c9a3e7bee3)

<p>算 dz：</p>

![QianJianTec1738852865655](https://github.com/user-attachments/assets/c7df7260-b3af-42cd-acb8-d91423704a5d)

<p>注：sigmoid 函数的导数等于 a(1 - a)</p>

<p>最后一步是反向算出你需要改变 w 和 b 多少</p>

![QianJianTec1738853160432](https://github.com/user-attachments/assets/95cf62f9-b8b4-4702-ac92-4613e4923a8f)

![QianJianTec1738853194658](https://github.com/user-attachments/assets/91f69274-c92e-4362-ab68-554a269838fd)

<p>所以如果要对一个例子进行梯度下降，就要算出 dz，再算出 dw 和 db，然后进行更新</p>

<p>以上是对于一个单一的训练样本如何计算导数和执行逻辑回归的梯度下降，但训练一个逻辑回归模型，不止有一个样本，而是有 m 个</p>

</br>

# m 示例的梯度下降

</br>

<p>回顾代价函数 J(w, b)，关心的是他的平均值</p>

![QianJianTec1738913541385](https://github.com/user-attachments/assets/6bcf317b-1b95-4888-bb8f-742a770a4afe)

![QianJianTec1738913649454](https://github.com/user-attachments/assets/90675d01-4a5a-4e00-93ac-5189c57913b0)

<p>上节展示只有一个训练样本是如何计算导数，现在总的代价函数作为一个总和的平均值，总代价函数对 w 的偏导数也将是对所有单个样本的损失对 w 的偏导求和再求平均值</p>

![QianJianTec1738914143779](https://github.com/user-attachments/assets/be78dfba-abcf-44a7-852b-e6f1dbe09032)

<p>前面已经展示了在单个训练样本如何计算 dw_i，所以真正做的是求得整体的梯度用于实现梯度下降，这里面有许多细节，但让我们把这一切看作一个具体的算法，这样你就可以实现逻辑回归的梯度下降</p>

</br>

# what can we do

</br>

```
J = 0, dw_1 = 0, dw_2 = 0, db = 0 //初始化

for i = 1 to m //在训练集中使用一个 for 循环，并计算每个训练样本的导数，然后相加
        z_i = w^T x_i + b
        a_i = sigma(z_i)
        J += -[y_i log(a_i) + (1 - y_i)log(1 - a_i)]
        dz_i = a_i - y_i
        dw_1 += x1_i * dz_i
        dw_2 += x2_i * dz_i
        db += dz_i

J /= m
dw_1 /= m, dw_2 /= m, db /= m

w_1 := w_1 - alpha * dw_1
w_2 := w_2 - alpha * dw_2
b := b - alpha * db
```

<p>该实现方法有缺点：要以这种方式实现逻辑回归，需要写两重循环</p>

1. 第一个 for 循环是用于在 m 训练样本上循环的

2. 第二个 for loop 用于计算这里的所有特征（因为可能出现 n 个特征，dw1，dw2，dwt）。
   
<p>当你在实现深度学习算法时，<b>有循环在代码里，会降低算法的运行效率</b>。在深度学习领域，会转移到一个越来越大的数据集，因此能够实现你的算法，而无需显示使用循环很重要，这有助于扩展大更大的数据集</p>

<p>在深度学习时代，使用矢量化摆脱显示使用循环变得很重要</p>



























