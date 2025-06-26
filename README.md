# Job_Shop_Scheduling
a two-stage hierarchical RL method framework of a variant of the job shop scheduling problem

## RL
an adaptive time window schedule and a rule schedule 

run on PyTorch 2.4.1 and Python 3.9


该方法基于分层强化学习范式，构建双层智能网络以优化弹药实时保障过程：上层智能体通过动态调整时间窗，根据系统当前状态自适应设置灵活窗口；下层智能体基于预定规则，自适应选择最优匹配策略，实现有限资源的合理分配。具体而言，上层智能体首先综合分析实时弹药保障态势和弹药保障机床运行状态，精准选择时间窗大小。通过将该时间窗内空闲弹药的决策信息传递至下层智能体，下层智能体将为空闲弹药选择最适合的启发式匹配规则，从而实现弹药保障的动态优化和资源高效利用。
![framework](framework.png "framework")

本小节主要讲述上层智能体的主要构建方式，如图所示，上层智能体参考Transformer，由编码器和解码器的的组件构成，其中，编码器的主要功能是处理弹药保障信息，通过对舰载机弹药种类、数量以及保障状态的综合分析，将这些原始数据转化为可用于后续计算的隐变量嵌入（embedding）。解码器的主要功能是根据输入的隐变量嵌入（embedding），通过对其进行处理和解码，输出相应的时间窗大小。通过这种方式，解码器则负责根据编码器生成的隐变量嵌入，对其进行进一步处理和解码，最终输出与任务时间窗大小相关的决策信息，以优化调度过程。编码器和解码器在结构上具有较高的一致性，主要由多头注意力机制（Multi-Head Attention, MHA），残差连接及归一化（Add \& Norm），前馈神经网络（Feed Forward）组成，我们以编码器为示例介绍嵌入传递过程，编码器由$L_1$个相同的注意力层的堆叠组成, 每一个注意力层由多头注意力层和前馈神经网络层(Feed Forward)组成, 每个操作后都会进行残差连接和归一化(add＆norm), 嵌入在模型中更新遵循两步过程。
![net](net.png "net")

