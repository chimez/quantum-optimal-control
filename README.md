# 切换至python3
# GRAPE-Tensorflow

这是我们最近出版的论文的代码仓库 "Speedup for quantum optimal control from automatic differentiation based on graphics processing units" https://journals.aps.org/pra/abstract/10.1103/PhysRevA.95.042318

这是使用 [Tensorflow] (https://www.tensorflow.org/) 的自动微分和GPU支持能力来运行量子优化控制的软件包. 主要目标是产生一组优化的脉冲,用来在给定的时间周期内驱动量子系统实现某一种幺正门或者达到某种末态,而且要保证保真度尽可能一致. 此外,用户可以添加任何的价值函数(处罚)在控制脉冲或量子中间态上,而且代码会自动将这个部分包括进优化过程,而无需写这个新的价值函数的梯度的解析形式.    

作为包产生结果的例子,这个是qubit pi脉冲的输出:  


![Qubit Pi Pulse Example](http://i.imgur.com/OfqFqZ6.png)

## 配置
只需要配置Tensorflow, Please follow the instructions [here] (https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html)  

Linux系统Python 3

## 目前实现的价值函数
参见 [Regularization functions file] (https://github.com/SchusterLab/GRAPE-Tensorflow/blob/master/core/RegularizationFunctions.py) 或者添加一个新的价值函数
 1. 保真度(fidelity): 目标幺正变换(或末态)与产生的幺正变换(或末态)间的重叠. 在代码里是 tfs.loss
 2. 高斯波包(gaussian envelope):如果控制脉冲没有高斯波包的惩罚. 用户在输入选项 `reg_coeffs` 中提供一个系数 `envelope`. 经验上0.01是比较好的值.
 3. 一阶导数(first derivative):让控制脉冲平滑. 用户在输入选项 `reg_coeffs` 中提供一个系数 `dwdt`. 经验上0.001是比较好的值.
 4. 二阶导数(second derivative):让控制脉冲平滑. 用户在输入选项 `reg_coeffs` 中提供一个系数 `d2wdt2`. 经验上0.000001是比较好的值.
 5. 带通(bandpass): 过滤控制脉冲频率 `bandpass` (从0.1左右开始)压制控制脉冲频率小于界限`band`. 这个价值函数需要GPU,因为TensorFlow QFT只在GPU里有.
 6. 禁止态价值函数: 这是一个在整个时间控制范围内禁止量子占据某个态的价值函数用户提供一个参数 `forbidden` (经验上从100开始)和一个列表`states_forbidden_list` 来指定禁止的能级的序号.  `forbid_dressed`:一个布尔值(默认为真)禁止在耦合系统中缀饰(哈密顿量的本征向量)裸态.
 7: 时间优化价值函数: 如果用户想加速门,他应该提供一个参数`speed_up`(从大约100开始)来奖励在所有中间态里的目标态,因而让门尽可能快.   


 增加一个新的价值函数:
只要按照跟我们用的一样的逻辑增加新代码[here](https://github.com/SchusterLab/GRAPE-Tensorflow/blob/master/core/RegularizationFunctions.py)
惩罚特性:
1) 控制场:在`tfs.ops_weight`里
和/或
2) 中间态:在`tfs.inter_vecs`里     
 

## 使用
 你应该调用这个函数:
```python
uks, U_final = Grape(H0,Hops,Hnames,U,total_time,steps,states_concerned_list,convergence, U0, 
reg_coeffs,dressed_info, maxA ,use_gpu, draw, initial_guess, show_plots, H_time_scales, 
unitary_error, method,state_transfer, no_scaling, freq_unit, file_name, save, data_path) 
```
 
 你可以按照[示例](https://github.com/SchusterLab/GRAPE-Tensorflow-Examples/tree/master)我们提供的详细信息关于定义量子体系，然后调用函数。我们建议从一个简单的例子开始(例如 spin Pi)。

## 返回值：
 `uks`: 优化的控制脉冲(一个floats列表的列表,其中每一个长度都等于`ctrl_steps(ctrl_op)`)与输入顺序相同.
 `U_final`:末态演化算符(nxn)
 
## 必须参数：
 `H0`: 漂移哈密顿量(nxn)
 `Hops`: 控制哈密顿量的列表(k个哈密顿量,每个都是nxn)
 `Hnames`: 控制哈密顿量的名字的列表,k个string元素
 `U`:目标演化算符(nxn)如果`state_transfer=False`一个矢量(nx1)如果`state_transfer=True`
 `total_time`: 总时间(float)
 `Steps`: 时间片段的数量(int)
 `states_concerned_list`: 定义初态(初态为列表,其中包括需要控制的态,可以有多个,格式为`[np.array,np.array]`)或初始演化算符(nxn矩阵)
 
## 可选参数：
 `U0`: 初始演化算符(nxn),默认是单位算符
 `convergence`: 一个字典(可以是空的),可能包括下面的参数及其默认值
                convergence = {'rate':0.01, 'update_step':100, 'max_iterations':5000,
                'conv_target':1e-8,'learning_rate_decay':2500, 'min_grad': 1e-25}
 `initial_guess`: k个元素的列表,每个元素是步骤大小向量,定义了每个算符的初始脉冲. 如果不提供的，默认值为高斯随机分布。
 `reg_coeffs`: 正规化参数的字典,这里定义了使用哪些价值函数
 `dressed_info`:包括有缀饰态本征值和本证向量的字典
 `maxA`:控制脉冲的最大振幅(默认值为4)
 `use_gpu`:布尔值:切换使用GPU和CPU,默认为真
 `sparse_H, sparse_U, sparse_K`:布尔值,指定(哈密顿量,幺正算符,幺正演化)是否是稀疏矩阵如果相应的稀疏度合适会加速计算(仅在CPU可用)
 `use_inter_vecs`:布尔值,开启/关闭图构建中态演化的参与
 `draw`:列表,包括在绘制态占据是包括的序号和名字Ex: `states_draw_list = [0,1], states_draw_names = ['g00','g01'] ,draw = [states_draw_list,states_draw_names]`
       默认值是绘制序号为0-3的态
 `show_plots`:布尔值(默认为真)在进度条和图表之间切换
 `state_transfer`:布尔值(默认为假)如果为真,目标为态转移如果为假,目标为幺正变换如果为真，`U`预计将是一个矢量，不是一个矩阵。
 `method`: 'ADAM', 'BFGS', 'L-BFGS-B' or 'EVOLVE'. 定义优化器默认是 `ADAM`. `EVOLVE` 只能模拟没有优化的传播.
 `Unitary_error`:float,指出想要的e指数的泰勒展开的最大误差来选择合适的展开项数,默认为`1e-4`
 `no_scaling`:布尔值(默认为假)禁用缩放和平方
 `Taylor_terms`:列表`[展开项,缩放和平方项]`,例如`[20,0]`,手动选择矩阵e指数的泰勒展开项
 `freq_unit`:字符串,默认为`GHz`可以是 'MHz', 'kHz' or 'Hz'
 `file_name`:保存模拟结果的文件名
 `save`:布尔值(默认为真)保存每一个新的步骤的控制算符,中间向量,末态演化
 `data_path`:保存模拟结果的路径
 
## 更多的例子：
我们用优化器在电路量子电动力学系统中产生光子薛定谔猫态： 
![光子薛定谔猫态](http://i.imgur.com/ponY2R9.png)
 

## 问题
如果有问题,请联系包的开发者: Nelson Leung (nelsonleung@uchicago.edu), Mohamed Abdelhafez (abdelhafez@uchicago.edu) or David Schuster (david.schuster@uchicago.edu)
