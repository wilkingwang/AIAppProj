### Function Calling是什么
Function Calling技术也被称作外部函数调用技术，也称为tool call。其核心用途是让LLM通过调用一些外部工具来完成工作。

### Function Calling执行流程
以天气查询为例，Function Calling流程：
<div align=center><img src="./images/Function Call/Function-Calling流程.png"></div>

大模型调用外部工具基本思路：
<div align=center><img src="./images/Function Call/LLM调用Function Calling流程.png"></div>
首先，调用外部工具的过程是根据用户意图触发的，而不是根据规则触发的。
此外，哪怕大模型挑选了合适的工具，作为语言模型，大模型最多只输出字符串，也就是说，大模型本身不具备调用外部工具的能力。为了解决这个问题，大模型采用一套非常特殊的模型训练流程，给大模型赋予识别外部工具的能力。
<div align=center><img src="./images/Function Call/Function-Calling原理.png"></div>

### Function Calling机制
无论Agent开发框架还是MCP技术，本质上都是Function Calling流程的效率方面的优化。
首先，Function Calling的核心是创建一个外部函数来连接大模型和外部工具API，这个外部函数至少需要两部分构成，其一是函数说明（输入和输出），其二是具体调用外部工具API的代码。
<div align=center><img src="./images/Function Call/Function-Calling机制.png"></div>
然后，模型会进行意图识别，若是无关问题，则会直接回答
<div align=center><img src="./images/Function Call/Function-Calling无关回答.png"></div>
而如果问题和外部函数相关，大模型会想外部函数发送一条Function Call Message函数调用消息，其中包含了需要调用的函数名称和运行函数所需参数。
<div align=center><img src="./images/Function Call/Function-Calling外部调用.png"></div>
最后，当外部函数接收到函数调用消息消息时，就会自动带入参数并运行，同时将其封装为一个function response message，加入原始消息列表，并让模型进行最终回复。

### LLM如何选择Function Calling
大模型的Function Calling就是用特殊标记符来规范的一种特殊响应模式。 当需要调用外部工具时，需要在系统提示词中添加关于外部函数的核心信息，包括函数名称、函数的输入和输出以及函数描述，从而引导模型让其适时的去调用外部工具。

当然，除此之外，输入的文本中肯定还包含一段通过特殊字符标记的用户输入的问题。而当模型接收到了这样的文本输入，就可能出现两种响应模式，其一是Function calling响应，也就是去创建那条调用外部函数的消息，此时表面上我们看到的消息是一个结构化文本，而真实的模型响应文本如下所示：

<div align=center><img src="./images/Function Call/模型调用Function-Call时输出.png"></div>

此时模型会通过大量的特殊标记符，来规范模型的输出，也就是此时模型只能输出调用函数的名称，如get_weather，以及对应的参数，如Beijing，而这就构成了咱们前面所说的Function call message。

而如果模型发现用户输入的问题和外部函数无关，比如用户输入“你好，好久不见”，那大模型就会按照如下方式进行响应。此时就没有那么多和tool相关的关键词了，很明显，这就是一个简单的一个文本的回应。

### Function Calling与模型能力关系
大模型是如何同时拥有这两种截然不同的响应模式（需要调用外部工具和不需要调用外部工具）？这是通过训练过程中指令微调方法，让模型能够有多种不同的响应方法。

一般来说，模型在训练过程会分模型训练和模型指令微调两个阶段：
- 预训练阶段：会带入海量文本，训练模型基础语言能力
- 指令微调：带入大量带有标签的文本数据，让模型学会各种问题该如何回答
除此之外，有些模型的训练过程中，还会在系统提示词中加入一些外部工具信息，并在output字段中加入function call message的创建信息。

通过这些数据的长期训练，就能让模型同时掌握对话能力和调用外部工具能力。同时，由于大模型具有举一反三的能力，因此在实际使用过程中遇到新的外部工具，大模型也能对其进行调用。


### Function Calling与API关系

### 提升Function准确性（https://developer.volcengine.com/articles/7390576761634816041#heading11）

#### 嵌套结构参数拉化

#### 添加系统提示词

#### 优化系统提示词

#### 在系统提示词中添加工具描述摘要

#### 优化函数名

#### 优化函数描述

#### 优化函数参数描述

#### 函数描述添加示例

#### 参数值添加示例

### 学习线路
- 选择业务场景
- 选择合适方案
- 实施