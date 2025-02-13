# Neural Networks Overview

</br>

## 回顾

</br>

<p>我们讨论过了逻辑回归，并且了解了这个模型和下面的流程图的关系</p>

![125b3fb2-cc31-41b2-9152-0354c37f4ac0](https://github.com/user-attachments/assets/50c9b46d-2b02-4d4d-affc-4697303f6b7b)

<p>在这里你需要输入特征 x，参数 w 和 b。用来计算 z，然后用 z 计算出 a，用 a 同时表示输出 yhat，之后你可以计算损失函数 L</p>

</br>

## Overview

</br>

<p>神经网络看起来是这样</p>
 
![bf092ee9-c0c6-4877-95c6-7db89d7975d2 13 30 07](https://github.com/user-attachments/assets/39d2c43c-a053-4717-9708-a500706a623f)

<p>之前提到过，可以将许多 sigmoid 单元（节点）堆叠起来构成一个神经网络</p>

<p>而之前，这个节点（单元）对应两个计算步骤：首先计算出 z，然后计算 a 值。</p>

<p>在这个神经网络中，这些节点将会对应一个类似 z 的计算（如上上图），同时对应一个类似 a 的计算（如上上图），然后其后面一层的节点会对应另一个类似 z 和 a 的计算</p>

<p>因此介绍一些新记号：首先用输入特征 x，以及参数 w[1] 和 b[1]，可以计算出 z[1]，所以引入一些新记号，使用上标 [1] 来表示与这些节点相关的量，也就是所谓的层</p>

![ef8b948e-74e7-43dd-8376-317e96ff35fa](https://github.com/user-attachments/assets/89703c08-0cbe-464c-a938-1561657422c3)

<p>之后使用上标 [2] 来表示与这个节点有关的量，这是神经网络的另一层</p>

![c10b70bd-3424-4e8f-83fb-58c99463ee24](https://github.com/user-attachments/assets/eeb8aa3e-1a04-448f-985b-5492102afacf)

<p>使用方括号，是为了不和圆括号混淆，我们使用圆括号来表示单独的训练样本（如 x^(i)），所以上标 (i) 表示第 i 个训练样本。上标 [1] 和 [2] 表示这些不同的层（神经网络的第一层，第二层）</p>

<p>使用类似逻辑回归去计算 z^[1]，然后使用 sigmoid(z^[1]) 来计算 a^[1]，<b>之后会用另一个线形方程计算 z^[2]</b>，然后计算 a^[2]。a^[2] 是整个神经网络的最后输出（用 yhat 表示）</p>

![f65a5d0a-9619-49aa-bddd-47a05cab5667](https://github.com/user-attachments/assets/bc43775d-b783-4ab7-9600-980d35c66815)

<p>这里面有很多细节，但要先学到这种直觉：<b></b>从一个逻辑回归中，先得到 z，后计算 a。</b>在上面的神经网络中，我们要做多次计算，反复计算 z 和 a，<b>最后计算损失函数</b></p>

<p>在逻辑回归中还会有反向计算，来计算导数 da dz。同样在神经网络中，也有类似的反向计算，按照上图从右向左计算</p>

</br>

# 神经网络表示法（Neural Network Representation）

</br>

<p>具体讨论神经网络图中的内容，从集中于神经网络的单隐藏层案例开始</p>

</br>

## One hidden layer Neural Network

</br>

<p>这是一张神经网络的图</p>

![69d8df4f-ddd5-47e7-95d7-f600158dafad](https://github.com/user-attachments/assets/6dd7f7f7-d87a-4ec0-8ce6-1eedb84fe559)

 <p>有输入 x1，x2，x3 垂直堆叠，也就是神经网络输入层(Input layer)</p>

<p>下一层叫做神经网络隐藏层(Hidden layer)，最后一层就只是一个节点成为输出层(Output layer)负责生成估计值 yhat</p> 

<p>在神经网络中，带有监督学习的训练集包含输入的 x，以及输出的 y。所以术语‘隐藏层’是指在训练集中，中间这一层节点的真实值并没有被观察，看得见输入输出，但训练集中间隐藏层的东西是看不见的</p>

<p>A stands for activations(激活)，指的是不同层次间的值，从神经网络的高层传递到下一层</p>

<p>在深度学习中，激活函数（activation function） 是神经网络中每个神经元的核心部分，它决定了神经元的输出。激活函数的作用是引入非线性特性，使得神经网络可以处理复杂的任务，如分类、回归、图像识别等。</p>

<p>所以输入层把 x 值传递到隐藏层,所以 X 也是activation</p>

![QianJianTec1739264114944](https://github.com/user-attachments/assets/be716a89-524c-437a-a071-987c9b0e5dc4)

<p>下一层的隐藏层将生成一些激活，称为 a^{[0]}</p>

<p>所以神经网络图中四个节点分别为 a^1_1，a^1_2，a^1_3，a^1_4</p>

![QianJianTec1739264648061](https://github.com/user-attachments/assets/700148c5-2e40-4d28-9845-450b78ec501d)

<p>所以 a^1 是个四维向量(在这个隐藏层有四个隐藏 units)</p>

<p>最后，在输出层再生成一些 a^2 值，这是一个实数，所以 yhat 就承接了 a^2 的值</p>

<p>所以这里的回归我们有 yhat 等于 a。</p>

<p>而逻辑回归我们只有一个输出层，所以不用上标，但在新网络，要使用上标来指明来自哪一层</p>

<p>a funny thing：这看到的网络是一个两层的神经网络，原因就是在神经网络计算分层时，不包括输入层，所以隐藏层是第一层，输出层是第二层</p>

<p>隐藏层会有相关参数 w 和 b，写上标方括号 1，来表示这些（w^1, b^1）是隐藏层的相关参数。</p>

<p>w 是一个 4*3 的矩阵，b 是一个 4*1 的向量。w 第一个维度 4 来自于隐藏层的四个节点和一个层，而 3 来自三个输入特征</p>

<p>在输出层的一些相关参数：w^2, b^2，维度分别为 1*4 和 1*1。1*4 是因为隐藏层有四个隐藏单位，输出层已经是一个单位</p>

</br>

# 计算神经网络的输出

</br>

<p>讨论神经网络计算它的输出的细节，会发现只是将逻辑回归进行多次的重复</p>

![69d8df4f-ddd5-47e7-95d7-f600158dafad](https://github.com/user-attachments/assets/6dd7f7f7-d87a-4ec0-8ce6-1eedb84fe559)

<p>这是一个双层的神经网络，深入了解他计算了什么</p>

<p>在逻辑回归中，一个圆代表了两步计算</p>

![29060c48-9def-4b57-bdaa-754cbb5ce1bc](https://github.com/user-attachments/assets/8346f261-de09-4de4-b14c-2bb29fa44357)

<p>先计算 z，然后计算 sigmoid(z) 作为激活函数</p>

<p>神经网络只是把这个过程做了多次</p>

<p>先看隐藏层的第一个节点，和逻辑回归类似，隐藏层中的节点进行了两步运算</p>

<p>首先计算了 z，使用的符号是关联了第一隐藏层中的所有节点的输入特征</p>

![QianJianTec1739275618543](https://github.com/user-attachments/assets/b7fd3f34-dbfa-4f81-86a7-107054e9e2f5)

<p>第二步计算 a_1^[1] = sigmoid(z_1^[1])</p>

<p>一个节点执行了这两步计算，剩下的第一层节点类推</p>

![93496f0a-de29-4b87-b91c-67ce75e21bf4](https://github.com/user-attachments/assets/b2b86ef6-b96a-49c4-88f6-42d496b70249)

<p>如果世纪实现一个神经网络，使用 for 循环来实现它似乎效率很低，所以接下来将这四个等式向量化</p>

<p>将从如何用向量的方法计算 z 开始：将这些 w（先转置成行向量） 叠放到一个矩阵中，得到了：</p>

![QianJianTec1739282050859](https://github.com/user-attachments/assets/a95888fb-4d4f-420a-bfbd-6eff8dfc49b0)

<p>也可以从另一个角度来解释：现在有四个逻辑回归单元，每一个逻辑回归单元都有一个相对应的参数向量 w，通过堆叠这四个向量，就得到了这个 (4, 3) 矩阵 W^[1]</p>

<p>所以如果用这个矩阵去乘输入变量</p>

![QianJianTec1739283962866](https://github.com/user-attachments/assets/1504d7c9-c411-4008-89e7-2f6ce3e93652)

<p>结果中，每一行都是一一对应，把这一大坨东西叫做向量 Z^[1]，就是把这些单独的 z 堆叠在一起而形成的列向量</p>

<p>在进行向量化的时候，有一个经验 is helpful：在一层中有不同的神经元时，就把他们堆叠起来。这就是为什么，但你有 z^[1]_1 到z^[1]_4 这些在隐藏层中对英语不同神经元的时候就把他们堆叠起来形成 Z^[1 ]</p>

<p>现在已经通过向量矩阵计算了 Z，最后要计算 a。显然会通过堆叠来定义 a。</p>

![QianJianTec1739284380658](https://github.com/user-attachments/assets/576ab467-df7d-413f-b5fa-e0832fe6b003)

<p>a^[1] 是 sigmoid 函数作用在 Z^[1] 上后得到的结果</p>

<p>sum up：</p>

![08614afe-a458-418a-84f1-d8573b123ef5](https://github.com/user-attachments/assets/e224db14-03cf-4774-9834-23b04e733b64)

<p>既然 x 可以用 a^[0] 代替，那么通过类似的推导，可以得出下一层的表达式</p>

<p>输出层的参数 w[2] 和 b[2] 分别是 (1, 4) 和 实数，z[2] 也是一个实数。（第二层和逻辑回归很相似）</p>

<p>接下来会看到，作用于多样本时，通过把训练样本堆叠在矩阵的不同列中，只需要通过很小的改变，就能将逻辑回归总的向量化实现照搬过来</p>

</br>

# Vectorizing across multiple 

</br>

<p>讨论如何对多个训练实例矢量化。结果会和逻辑回归的结果很相似，通过把不同的训练实例按列堆叠在一个矩阵中，就可以拿之前的等式。做一些很小的改动，使得神经网络同时计算所有训练实例的预测</p>

![15fdc900-7b6b-44d2-82ca-4da904632c0d](https://github.com/user-attachments/assets/61cac5dd-c9e1-495a-a6d0-9205eae9a6ec)

<p>这是之前的四个等式，这些等式告诉我们，给定一个特征矢量 x，可以用来取得一个训练实例的 yhat。如果有 m 个训练实例，需要重复这个过程（x^m 用来预测 a[2]^m = yhat^m）</p>

<p>非矢量化实现：</p>

![b47167c1-76b1-430d-92d3-c7acf338922b](https://github.com/user-attachments/assets/4b85b8cf-b2a9-4af3-bfd9-795e13d4c39c)

<p><b>向量化实现：</b></p>

![QianJianTec1739353499704](https://github.com/user-attachments/assets/7e19fb91-8d09-4c64-88fb-64c98a6261c3)

![QianJianTec1739353558058](https://github.com/user-attachments/assets/7d55a15d-03bd-4b25-8634-c9a0a98aaf99)

![QianJianTec1739353634811](https://github.com/user-attachments/assets/913d167d-165d-40d4-99ac-fd17b514bbc6)

![QianJianTec1739353682402](https://github.com/user-attachments/assets/5445e245-82c3-45db-bf1c-40fa7f41cde0)

<p>找相似对比，从 x 出发，把 x 堆叠得到 X。如果对 z 做同样的操作（注意这些也是列向量）得到 Z[1]</p>

![QianJianTec1739354728733](https://github.com/user-attachments/assets/da48c355-f378-46d1-8e33-191ae84fe660)

<p>A[1] 也同理</p>

![QianJianTec1739354825339](https://github.com/user-attachments/assets/ed520c75-3f0c-4522-b677-16e0a95d16a1)

<p>对于 Z[2] 和 A[2] 也是一样的横着堆叠。横向来看 A 代表不同训练实例，纵向来看代表不同的隐藏单元</p>

</br>

# 矢量化实施的解释

</br>

<p>给出说明，确保所写下的公式是对多个样本向量化的正确实现</p>

<p>看一下对于几个样本前向传播计算过程的一部分</p>

![QianJianTec1739360980378](https://github.com/user-attachments/assets/3c0d3a1a-62ec-4f0f-ab84-13a70fb32111)

![QianJianTec1739360953150](https://github.com/user-attachments/assets/6c0e999a-ef01-4141-ab05-e401865aab10)

<p>为了简要说明：将省略 b</p>

<p>考虑 w[1]x(i)（对于每一个训练实例都是如此）</p>

![QianJianTec1739371569008](https://github.com/user-attachments/assets/5ec74e60-1aa8-456a-8fa9-4928f5445d3e)

<p>现在，如果将训练集记成大写的 X，他是通过将所有的训练样本堆叠在一起得到的，X 的构成是在向量 x 之上，依次水平堆叠 x1，x2，x3，...</p>

<p>将 W 与这个矩阵相乘，将会得到水平堆叠的 w[1]x(1),即 Z[1]</p>

<p>把 X 改成 A[0] 后，公式12与公式34很相似，实际上说明了<b>一个神经网络的不同层大致上都在做着相同的计算</b></p>

</br>

# Activation functions

</br>

<p><b>激活函数用于引入非线性，使模型能够学习复杂的映射关系。如果没有激活函数，神经网络只是多个线性变换的叠加，无法拟合复杂的非线性问题。</b></p>

<p>当你构建神经网络时，你要做出的选择之一就是在隐藏层以及神经网络输出单元中使用什么激活函数</p>

<p>到目前为止，我们的神经网络一直在用 sigmoid 函数，这实际上并不是最好的选择。讨论该如何使用不同的激活函数</p>

![cb1b197d-8603-4f46-a8ab-a2c787e97356](https://github.com/user-attachments/assets/42cba352-bbf4-4e0e-8749-4db974678871)

<p>在神经网络的前向传播中，有两个步骤使用 sigmoid 函数。在更一般的情况下，我们可以使用不同的函数--g(z) 可能不是 sigmoid 的非线性函数。</p>

<p>几乎总是比 sigmoid 函数更好的激活函数是双曲正切函数（tanh(x) 作为激活函数用于神经网络，能将输入数据映射到 (-1,1) 之间，有平滑性和抗噪性。）从数学上讲，他实际上是 sigmoid 函数的移位版本</p>

![QianJianTec1739440952827](https://github.com/user-attachments/assets/b57e5f6b-ab4b-428a-a6df-815328a00517)

<p>事实证明，<b>对于隐藏单位</b>如果让 g(z) = tanh(z)，这几乎总是比 sigmoid 函数效果更好，因为值介于 (1, -1)，则来自隐藏层的激活的平均值更接近于零均值。因此，就像有时在训练学习算法一样，可以使用 tanh 而不是 sigmoid 将数据居中并是数据均值为零（而不是 0.5），这实际上使下一层的学习变得更加容易一些（当学习优化算法时，会讨论这个问题）<p>

<p>有一个收获是，我几乎再也不会使用 sigmoid 函数，严格来说，tanh 函数几乎总是更胜一筹，唯一的例外是输出层，因为如果 y 属于{0, 1}，那么 yhat 的输出介于 (0, 1) 才是有意义的。因此使用 sigmoid 函数的一个例外是使用二元分类，在这种情况下，可以对输出层使用 sigmoid 函数</p>

<p>所以在这个神经网络中隐藏层有一个 tanh 函数，输出层有 sigmoid 函数（g[1](z[1])，g[2](z[2])）</p>

<p>sigmoid 函数和 tanh 函数唯一的缺点是，如果 z 非常大或非常小，则该函数斜率的导数的梯度就会变得非常小（接近于零），因此这可能会减慢梯度下降的速度</p>

<p>因此，在机器学习中非常流行的另一种选择是所谓的整流线型单元(ReLU)，公式为：</p>

`a = max(0, z)`

<p>因此，只要 z 为正则导数为 1，z 为负则导数为 0。关于 z 等于 0 时导数是不明确的，当 z 为零时，你可以假装导数为一或零，仍可以很好的工作</p>

<p>事实上，有一些选择激活函数的经验法则：</p>

- 如果使用二元分类，如果值是 0 或 1，那么对于输出层来说，sigmoid 激活函数是非常自然的选择

- 对于其他 unit，整流线型单元越来越成为激活函数的默认首选。所以，如果不确定用什么来做隐藏层，就用 ReLU（尽管人们也会使用 tanh 激活函数），他的一个缺点是，当 z 为负时，导数等于零。

<p>ReLU 还有另一个版本叫做 Leaky ReLU。当 z 为负值时，Leaky ReLU 不为零，而是稍微倾斜一些，通常比 ReLU 效果更好，但是在实践中没有那么多使用。</p>

<p>ReLU 和 Leaky ReLU 比的优势在于：在 z 的大部分空间（激活函数的导数）中，激活函数的斜率与 0 有很大的不同</p>

<p>因此在实践中，使用 ReLU 激活函数，你的神经网络的学习速度通常比使用 tanh 和 sigmoid 快很多。原因：函数的斜率变为 0 时的影响较小，这会减慢学习速度。而对于 z 范围的一半，值的斜率变为 0。但实际上，隐藏单位中有足够多的 z 会大于零。因此对于大多数训练实例来说，学习速度仍然非常快</p>

<p>sigmoid 除了二元分类几乎不用。tanh 要优越得多。然后是默认的最常用的激活函数 ReLU。</p>

<p>也许，也可以随意试试 Leaky-ReLU。</p>

![08de301d-2a4f-4d20-afc3-c42ef1bf5877](https://github.com/user-attachments/assets/a6ef33f5-250e-4a16-a15b-e50232c6ab01)

`a = max(0.01 * z, z)`

<p>至于为什么是 0.01，<b>也可以把它当作学习算法的另一个参数</b>，自己试试吧，如果能带来不错的效果，就坚持下去。</p>

<p>sum up：将在深度学习中看到的是，在如何构建神经网络方面，通常会有很多不同的选择，从许多隐藏单位到选择激活函数，再到如何初始化。事实证明，有时很难获得好的指导方针</p>

</br>

# 为什么需要激活函数

</br>

<p>为什么神经网络需要使用非线性激活函数，事实上想要神经网络实现有趣的功能，确实需要选择一个非线性激活函数</p>


















