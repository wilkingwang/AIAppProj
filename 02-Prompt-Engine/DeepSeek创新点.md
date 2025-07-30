## MLA(Multi-Head Latent Attention)
&ensp;&ensp;传统的多头注意力(Multi-Head Attention)的键值(KV)缓存机制事实上对计算效率形成了较大阻碍，缩小KV缓存(KV Cache)大小，并提高性能，在之前模型架构中并未很好的解决。
&ensp;&ensp;DeepSeek引入MLA，一种通过低秩键值联合压缩的注意力机制，在显著减小KV缓存的同时提升计算效率。低秩近似是快速矩阵计算的常用方法，在MLA之前很少用于大模型计算。

## MOE(Mix of Expert)
&ensp;&ensp;MoE架构的模型虽然总参数量很大，但每次训练或推理时只需要激活很少链路（分为公共专家Shared Expert和个性化专家Routed Expert，通过Router路由选择具体哪个专家），训练成本大大降低，推理速度显著提高。

## 混合精度框架
&ensp;&ensp;采用了混合精度框架，即在不同的区块里使用不同的精度来存储数据(精度越高，内存占用越多，运算复杂度越大。DeepSeek在一些不需要很高精度的模块中，使用很低的精度FP8存储数据，极大的降低了训练计算量)。

## DeepSeek-R1推理能力
&ensp;&ensp;DeepSeek-R1推理能力强大，是因为具有以下两点能力：
- 强化学习驱动
&ensp;&ensp;普通训练使用监督学习模式，DeepSeek-R1通过大规模强化学习技术显著提升了推理能力。
- 长链推理(CoT)技术
&ensp;&ensp;DeepSeek-R1采用长链推理技术，其思维链长度很长，能够逐步分解复杂问题，通过多步骤的逻辑推理来解决问题。

## 课后扩展
### 模型推理流程
&ensp;&ensp;LLM推理分为两个阶段：Prefill阶段和Decode阶段
- Prefill
&ensp;&ensp;是模型对全部的Prompt Tokens一次性并行计算，最终会生成第一个输出Token

&ensp;&ensp;在推理过程中，由于模型堆叠了多层transformer，所以核心的计算消耗在Transformer内部，包括MHA，FFN等操作，其中MHA要计算Q，K ，V 矩阵，来做多头注意力的计算。

公式中的符号：$t$表示计算序列中第$t$个token；$q,k,v,o$中的两个下标，前一个表示token位置，后一个表示对应的Head下标。

从从公式 （$7$）可以看到，在计算Attention时，$t$位置的$q$只与&t&位置前的$k, v$做计算，所以我们有如下两个结论：
1、计算前面的$k, v$并不受后面token的影响。
2、后面计算$t+1、t+2、...，t+n$位置的Attention，要使用前序的$1 -> t$位置的$k, v$的值是始终不变的。

- Decode
&ensp;&ensp;每次生成一个Token，知道生成EOS(End-of-sequence) Token，产生最终的response

### 多头注意力机制

### KV Cache
为什么显著减小KV Cache，计算效率能提升

### 低秩近似

### 