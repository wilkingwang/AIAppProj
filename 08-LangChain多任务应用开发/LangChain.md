### LangChain
Prompt、LLM、Memory、Chain

#### Models
模型的管理

#### Prompt
提示词管理，包括提示词管理、提示词优化和提示词序列化

#### ReAct模式
Reasoning and Actioning就是在推理的过程中分步骤给出结果，在原有的基础上

- 为什么需要对思考限制进行限制

- fewshot与ReAct区别

- 什么样的场景适合ReAct模式

- 实现原理

#### 工具

#### 记忆
保存和模型的交互式的上下文

- BufferMemory

- BufferWindowMemory

- ConversionMemory
将对话进行摘要，将摘要存储在内存中，相当于将压缩过的历史对话传递给LLM

- VectorStore-backed Memory
将之前所有对话通过向量存储到VectorDB中，每次对话，会根据用户的输入信息，匹配向量数据库中最相似的K对话

#### 知识库Indexes
用于结构化文档，方面和模型交互。如果要构建自己的知识库，就需要各种类型文档的加载、转换、长文本切分、文本向量计算、向量索引存储查询等。

#### Chain
一些列

- ConversationChain
https://python.langchain.com/api_reference/langchain/chains/langchain.chains.conversation.base.ConversationChain.html

#### 任务链

#### Agent

##### Agent类型
- zero-shot-react-description：根据工具的描述和请求内容来决定使用哪个工具
- react-docstore：使用ReAct框架和docsotre交互使用Search或Lookup工具，前者用来搜索，后者寻找term
- convertsational-react-description：为会话设置而设计的代理，他的prompt会被设计的具有会话性，且还是会使用ReAct框架来决定使用哪个工具，并且将过往的会话交互存入内存

### LangGraph


### LangMath


### 问题
- ReAct与LLM Thinking有何区别

- ReAct与CoT关系
CoT是思维链，是事先确定好的流程，

- 工具识别准确度与什么有关系？模型大小？工具描述？