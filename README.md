# deepkoopman
参考论文A Data-Efficient Reinforcement Learning Method Based on Local Koopman Operators的koopman方案实现

## 1. 环境
需要pytorch，numpy

## 2. 运行与结果
运行`python main.py`即可，可以修改env
会间隔输出拟合结果对比，最终输出一个test_error的图

## 3. 代码结构
- `main.py`：主程序，包括生成初始轨迹，调用训练，与训练结果测试
- `deep_util.py`：训练程序包括网络定义，replaybuffer定义，训练定义
- 'nonlinear_system.py'：非线性系统的定义，含有VanDerPol，弹簧阻尼非线性系统，和强非线性的奖励
- 'test_all.py'：用pytest的单元测试