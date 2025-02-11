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

<p></p>































