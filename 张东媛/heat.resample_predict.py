"""
使用物理信息神经网络（PINNs）求解一维热传导方程
方程：∂u/∂t = a * ∂²u/∂x², 0 < x < L, 0 < t < 1
边界条件：u(0,t) = u(L,t) = 0
初始条件：u(x,0) = sin(nπx/L)
代码示例：基于DeepXDE库的PINNs实现
"""

"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
# 导入必要的库
import deepxde as dde  # DeepXDE: 用于物理信息神经网络的主要库
import numpy as np  # NumPy: 用于数值计算和数组操作


def heat_eq_exact_solution(x, t):
    """
    计算热传导方程的精确解（针对正弦初始条件）

    参数
    ----------
    x : np.ndarray
        空间坐标值
    t : np.ndarray
        时间坐标值

    返回
    ----------
    np.ndarray
        在给定(x,t)处的精确解值
    """
    # 一维热传导方程的解析解：u(x,t) = e^(-n²π²a t/L²) * sin(nπx/L)
    return np.exp(-(n**2 * np.pi**2 * a * t) / (L**2)) * np.sin(n * np.pi * x / L)


def gen_exact_solution():
    """
    生成热传导方程的精确解数据并保存到文件

    该函数：
    1. 在定义的时空域上创建网格
    2. 计算每个网格点上的精确解
    3. 将数据保存为.npz文件供后续使用
    """
    # 定义每个维度的点数：x方向256点，t方向201点
    x_dim, t_dim = (256, 201)

    # 定义x和t的取值范围：x ∈ [0, L], t ∈ [0, 1]
    x_min, t_min = (0, 0.0)
    x_max, t_max = (L, 1.0)

    # 创建时间向量t：从t_min到t_max，共t_dim个点，形状为(t_dim, 1)
    t = np.linspace(t_min, t_max, num=t_dim).reshape(t_dim, 1)
    # 创建空间向量x：从x_min到x_max，共x_dim个点，形状为(x_dim, 1)
    x = np.linspace(x_min, x_max, num=x_dim).reshape(x_dim, 1)
    # 创建存储精确解的零矩阵：形状为(x_dim, t_dim)
    usol = np.zeros((x_dim, t_dim)).reshape(x_dim, t_dim)

    # 遍历所有网格点，计算精确解
    for i in range(x_dim):
        for j in range(t_dim):
            usol[i][j] = heat_eq_exact_solution(x[i], t[j])

    # 保存数据到.npz文件：包含x、t和精确解usol
    np.savez("heat_eq_data", x=x, t=t, usol=usol)

    # 验证：加载保存的数据（可选）
    data = np.load("heat_eq_data.npz")
    print(f"精确解数据已保存，形状：x={data['x'].shape}, t={data['t'].shape}, usol={data['usol'].shape}")


def gen_testdata():
    """
    导入并预处理包含精确解的数据集

    返回
    ----------
    X : np.ndarray
        测试数据点，形状为(N, 2)，每行包含(x, t)
    y : np.ndarray
        对应的精确解值，形状为(N, 1)
    """
    # 加载保存的精确解数据
    data = np.load("heat_eq_data.npz")

    # 提取t、x和精确解usol
    # 注意：usol需要转置以匹配网格格式
    t, x, exact = data["t"], data["x"], data["usol"].T

    # 创建网格：xx和tt分别是x和t在网格上的坐标矩阵
    xx, tt = np.meshgrid(x, t)

    # 将网格数据展平为二维数组：
    # np.ravel(xx)展平xx，np.ravel(tt)展平tt
    # np.vstack将两个展平的数组合并为2行N列的数组
    # .T转置为N行2列，每行是一个(x, t)点
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T

    # 将精确解也展平为一维数组，然后转换为列向量
    y = exact.flatten()[:, None]

    return X, y


# ==================== 问题参数设置 ====================
a = 0.4   # 热扩散系数：控制热传导速度，值越大传导越快
L = 1     # 杆的长度：空间域的长度
n = 1     # 正弦初始条件的频率：决定初始温度分布的波数

# ==================== 生成精确解数据 ====================
# 如果没有精确解数据文件，则生成
# 如果有现成数据，可以注释掉这一行以节省时间
gen_exact_solution()


def pde(x, y):
    """
    定义热传导方程的PDE残差

    参数
    ----------
    x : tensor
        输入变量，形状为(N, 2)，包含空间坐标x和时间坐标t
    y : tensor
        神经网络输出，形状为(N, 1)，表示预测的u值

    返回
    ----------
    tensor
        PDE残差，形状为(N, 1)：∂u/∂t - a * ∂²u/∂x²
    """
    # 使用DeepXDE的自动微分计算偏导数：

    # dy_t: 计算u对t的偏导数（∂u/∂t）
    # jacobian(y, x, i=0, j=1):
    #   - y: 输出变量
    #   - x: 输入变量
    #   - i=0: 对y的第0个输出求导（本例中y只有1个输出）
    #   - j=1: 对x的第1个输入求导（x的第0列是x，第1列是t）
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)

    # dy_xx: 计算u对x的二阶偏导数（∂²u/∂x²）
    # hessian(y, x, i=0, j=0):
    #   - 计算y对x的二阶导数
    #   - i=0, j=0: 表示对x的第0个输入求两次导
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)

    # PDE残差：∂u/∂t - a * ∂²u/∂x²
    # 当且仅当预测解满足PDE时，残差为0
    return dy_t - a * dy_xx


# ==================== 定义计算几何 ====================
# 空间几何：长度为L的一维区间
geom = dde.geometry.Interval(0, L)

# 时间域：从0到1的时间区间
timedomain = dde.geometry.TimeDomain(0, 1)

# 时空几何：空间几何和时间域的笛卡尔积
geomtime = dde.geometry.GeometryXTime(geom, timedomain)


# ==================== 定义边界条件和初始条件 ====================
# 边界条件：Dirichlet边界条件（第一类边界条件）
# 在边界上u=0
bc = dde.icbc.DirichletBC(
    geomtime,                    # 几何域：时空域
    lambda x: 0,                # 边界上的值：恒为0
    lambda _, on_boundary: on_boundary  # 选择函数：只选择边界上的点
)

# 初始条件：t=0时的温度分布
ic = dde.icbc.IC(
    geomtime,                    # 几何域：时空域
    # 初始值函数：u(x,0) = sin(nπx/L)
    lambda x: np.sin(n * np.pi * x[:, 0:1] / L),
    lambda _, on_initial: on_initial  # 选择函数：只选择初始时间点
)

# ==================== 定义回调函数 ====================
# PDEPointResampler：周期性重新采样PDE残差点
# 这有助于防止过拟合到特定的采样点，提高训练稳定性
pde_resampler = dde.callbacks.PDEPointResampler(period=10)


# ==================== 定义PDE问题 ====================
data = dde.data.TimePDE(
    geomtime,        # 时空几何域
    pde,             # PDE残差函数
    [bc, ic],        # 边界条件和初始条件

    # 训练点配置：
    num_domain=2540,    # 域内采样点数量（用于计算PDE残差）
    num_boundary=80,    # 边界采样点数量（用于边界条件）
    num_initial=160,    # 初始条件采样点数量

    # 测试点配置：
    num_test=2540,      # 测试点数量（用于验证）
)


# ==================== 构建神经网络 ====================
# FNN：全连接前馈神经网络
# 网络结构：[2] + [20]*3 + [1] = 输入层(2) → 隐藏层1(20) → 隐藏层2(20) → 隐藏层3(20) → 输出层(1)
net = dde.nn.FNN(
    [2] + [20] * 3 + [1],  # 网络层结构：输入2维，3个隐藏层每层20个神经元，输出1维
    "tanh",                # 激活函数：双曲正切（适合求解PDE）
    "Glorot normal"        # 权重初始化方法：Glorot正态分布（Xavier正态分布）
)

# ==================== 创建模型 ====================
model = dde.Model(data, net)


# ==================== 第一阶段训练：使用Adam优化器 ====================
# Adam优化器：自适应矩估计，适合初始阶段的快速收敛
model.compile(
    "adam",      # 优化器：Adam
    lr=1e-3      # 学习率：0.001
)

# 训练模型：200000次迭代
print("开始第一阶段训练（Adam优化器）...")
model.train(
    iterations=20000,      # 迭代次数
    callbacks=[pde_resampler]  # 回调函数：周期性重新采样PDE点
)


# ==================== 第二阶段训练：使用L-BFGS优化器 ====================
# L-BFGS优化器：拟牛顿法，适合精细调优，收敛速度快但内存消耗大
model.compile("L-BFGS")

print("开始第二阶段训练（L-BFGS优化器）...")
# 训练并获取训练历史和状态
losshistory, train_state = model.train(
    callbacks=[pde_resampler]  # 回调函数：周期性重新采样PDE点
)


# ==================== 结果保存与可视化 ====================
# 保存训练损失曲线和模型状态图
dde.saveplot(
    losshistory,      # 训练历史（包含损失值）
    train_state,      # 训练状态（包含模型参数）
    issave=True,      # 保存图片到文件
    isplot=True       # 显示图片
)


# ==================== 模型评估 ====================
# 生成测试数据
X, y_true = gen_testdata()

# 使用训练好的模型进行预测
y_pred = model.predict(X)

# 计算PDE残差：检查预测解在多大程度上满足PDE
f = model.predict(X, operator=pde)

# 输出评估指标
print("=" * 50)
print("模型评估结果：")
print(f"平均残差: {np.mean(np.absolute(f)):.6e}")
print(f"L2相对误差: {dde.metrics.l2_relative_error(y_true, y_pred):.6e}")
print("=" * 50)

# ==================== 保存详细结果 ====================
# 将测试数据、真实值和预测值保存到文本文件
np.savetxt(
    "test.dat",                    # 文件名
    np.hstack((X, y_true, y_pred)), # 数据：合并X, y_true, y_pred
    header="x t y_true y_pred",     # 文件头
    comments=""                     # 注释字符
)

print("结果已保存到 test.dat 文件")
print("=" * 50)
print("PINNs训练完成！")


# ==================== 添加：自定义点预测 ====================
def predict_temperature(x, t):
    """
    预测任意空间点x和时间t的温度

    参数
    ----------
    x : float or array
        空间坐标，可以是单个值或数组
    t : float or array
        时间坐标，可以是单个值或数组

    返回
    ----------
    float or array
        预测的温度值
    """
    # 确保输入是二维数组格式
    if np.isscalar(x) and np.isscalar(t):
        # 单个点预测
        input_point = np.array([[x, t]])
        return model.predict(input_point)[0, 0]
    else:
        # 多个点预测
        # 确保x和t有相同的形状
        x_array = np.array(x).reshape(-1, 1)
        t_array = np.array(t).reshape(-1, 1)
        input_points = np.hstack([x_array, t_array])
        predictions = model.predict(input_points)
        return predictions.flatten()


# ==================== 示例1：单个点预测 ====================
print("\n示例1：单个点预测")
x_test = 0.5
t_test = 0.3
u_pred = predict_temperature(x_test, t_test)
print(f"在 x={x_test}, t={t_test} 处的预测温度: {u_pred:.6f}")

# 计算精确解作为比较
u_exact = heat_eq_exact_solution(x_test, t_test)
print(f"在 x={x_test}, t={t_test} 处的精确温度: {u_exact:.6f}")
print(f"绝对误差: {abs(u_pred - u_exact):.6e}")

# ==================== 示例2：多个点预测 ====================
print("\n示例2：多个点预测")
x_points = [0.2, 0.4, 0.6, 0.8]
t_points = [0.1, 0.2, 0.3, 0.4]
print(f"空间点: {x_points}")
print(f"时间点: {t_points}")

predictions = predict_temperature(x_points, t_points)
for i, (x, t) in enumerate(zip(x_points, t_points)):
    exact = heat_eq_exact_solution(x, t)
    print(
        f"点 {i + 1}: x={x:.1f}, t={t:.1f} -> 预测={predictions[i]:.6f}, 精确={exact:.6f}, 误差={abs(predictions[i] - exact):.6e}")

# ==================== 示例3：时间序列预测（固定位置） ====================
print("\n示例3：在固定位置x=0.5处的时间序列")
x_fixed = 0.5
t_series = np.linspace(0, 1, 11)  # 11个时间点
u_series = predict_temperature([x_fixed] * len(t_series), t_series)

print(f"在 x={x_fixed} 处的时间序列预测:")
for t, u in zip(t_series, u_series):
    exact = heat_eq_exact_solution(x_fixed, t)
    print(f"  t={t:.2f}: u={u:.6f} (精确={exact:.6f})")

# ==================== 示例4：空间分布预测（固定时间） ====================
print("\n示例4：在固定时间t=0.5处的空间分布")
t_fixed = 0.5
x_series = np.linspace(0, L, 11)  # 11个空间点
u_distribution = predict_temperature(x_series, [t_fixed] * len(x_series))

print(f"在 t={t_fixed} 处的空间分布预测:")
for x, u in zip(x_series, u_distribution):
    exact = heat_eq_exact_solution(x, t_fixed)
    print(f"  x={x:.2f}: u={u:.6f} (精确={exact:.6f})")


# ==================== 可视化预测结果（可选） ====================
def visualize_predictions():
    """可视化预测结果"""
    try:
        import matplotlib.pyplot as plt

        # 创建网格
        x_vals = np.linspace(0, L, 50)
        t_vals = np.linspace(0, 1, 50)
        X_grid, T_grid = np.meshgrid(x_vals, t_vals)

        # 展平网格用于预测
        X_flat = X_grid.flatten()
        T_flat = T_grid.flatten()

        # 预测
        U_pred = predict_temperature(X_flat, T_flat)
        U_pred_grid = U_pred.reshape(X_grid.shape)

        # 精确解
        U_exact_grid = np.zeros_like(X_grid)
        for i in range(X_grid.shape[0]):
            for j in range(X_grid.shape[1]):
                U_exact_grid[i, j] = heat_eq_exact_solution(X_grid[i, j], T_grid[i, j])

        # 创建子图
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # 1. 预测解
        im1 = axes[0].contourf(X_grid, T_grid, U_pred_grid, levels=20, cmap='hot')
        axes[0].set_xlabel('space x')
        axes[0].set_ylabel('time t')
        axes[0].set_title('PINNs-predict-result')
        plt.colorbar(im1, ax=axes[0])

        # 2. 精确解
        im2 = axes[1].contourf(X_grid, T_grid, U_exact_grid, levels=20, cmap='hot')
        axes[1].set_xlabel('space x')
        axes[1].set_ylabel('time t')
        axes[1].set_title('precies-result')
        plt.colorbar(im2, ax=axes[1])

        # 3. 误差
        error = np.abs(U_pred_grid - U_exact_grid)
        im3 = axes[2].contourf(X_grid, T_grid, error, levels=20, cmap='viridis')
        axes[2].set_xlabel('space x')
        axes[2].set_ylabel('time t')
        axes[2].set_title('absolute error')
        plt.colorbar(im3, ax=axes[2])

        plt.tight_layout()
        plt.savefig('predictions_comparison.png', dpi=150)
        plt.show()

    except ImportError:
        print("注意：未安装matplotlib，跳过可视化")


# 运行可视化
visualize_predictions()
# ==================== 说明 ====================
"""
重要概念解释：

1. 物理信息神经网络（PINNs）:
   - 与传统神经网络不同，PINNs将物理方程（PDE）作为约束加入损失函数
   - 通过最小化PDE残差、边界条件残差和初始条件残差来训练网络
   - 不需要大量的标注数据，只需要PDE本身和边界/初始条件

2. 损失函数组成:
   - PDE损失: L_pde = mean(||∂u/∂t - a*∂²u/∂x²||²)
   - 边界条件损失: L_bc = mean(||u(边界点) - 0||²)
   - 初始条件损失: L_ic = mean(||u(t=0) - sin(nπx/L)||²)
   - 总损失: L_total = L_pde + L_bc + L_ic

3. 两阶段训练策略:
   - 第一阶段（Adam）: 全局搜索，快速降低损失
   - 第二阶段（L-BFGS）: 局部精细调优，达到更高精度

4. 网络架构选择:
   - 激活函数tanh: 平滑可微，适合表示光滑的PDE解
   - 3层隐藏层: 足够的表达能力拟合复杂函数
   - 每层20个神经元: 平衡表达能力和计算效率

5. 采样策略:
   - 域内点: 评估PDE残差
   - 边界点: 强制边界条件
   - 初始点: 强制初始条件
   - 测试点: 独立评估模型性能

常见问题及解决方案:
1. 训练不收敛: 尝试降低学习率、增加网络宽度/深度、调整采样点数量
2. 过拟合: 增加正则化、使用更简单的网络结构、增加训练数据
3. 梯度消失/爆炸: 使用合适的激活函数、权重初始化、梯度裁剪
4. 精度不足: 增加训练迭代次数、使用自适应采样、调整优化器参数
"""