# DiffCFD Rust 改造计划

> 基于 `ddfcfd-reform-1.txt` 五阶段方案制定。原则：不向后兼容，始终最优决策。

## 总体架构

- **前向求解 (Forward)**：纯物理计算，不参与 PyTorch 跟踪 → 下沉到 Rust
- **反向传播 (Backward)**：C1 隐式微分，需在收敛点评估残差 → 保留 PyTorch（vjp）或 Rust 显式雅可比
- **数据通道**：PyTorch Tensor → `.detach().numpy()` → PyO3 零拷贝视图 → Rust 计算 → NumPy 数组返回

---

## Phase 1: Rust 工具链与混合项目初始化

- [x] 1.1 安装 maturin
- [x] 1.2 创建 Rust crate 结构 (`src/` + `Cargo.toml`)
- [x] 1.3 配置 pyproject.toml 使用 maturin 构建后端
- [x] 1.4 引入依赖: pyo3, numpy, ndarray, rayon
- [x] 1.5 实现零拷贝桥梁: 基础 PyO3 函数暴露
- [x] 1.6 验证: `pytest tests/unit/` 通过 + Python 可 `import diffcfd`

## Phase 2: 稀疏矩阵组装加速

- [x] 2.1 Rust: 实现 `build_momentum_u` (CSR 格式)
- [x] 2.2 Rust: 实现 `build_momentum_v` (CSR 格式)
- [x] 2.3 Rust: 实现 `build_pressure_system` (CSR 格式)
- [x] 2.4 Python: NavierStokes2D 调用 Rust 版本替换 _solve_u/_solve_v/_build_pressure_system
- [x] 2.5 验证: Rust 计算结果与纯 Python 版本浮点误差 < 1e-7

## Phase 3: SDF 几何计算加速

- [x] 3.1 Rust: 实现 BSplineAirfoil 点到多边形距离计算 (rayon 并行)
- [x] 3.2 Rust: 实现射线投射 winding number 内外判定 (rayon 并行)
- [x] 3.3 Python: BSplineAirfoil.sdf 调用 Rust 版本
- [x] 3.4 验证: NACA0012 和随机 B-spline SDF L2 误差 < 1e-6

## Phase 4: 前向 SIMPLE 求解器完全下沉

- [x] 4.1 Rust: 使用 faer-sparse 实现稀疏线性求解
- [x] 4.2 Rust: 实现完整 SIMPLE 外循环 (动量步 + 压力修正 + 速度修正)
- [x] 4.3 Python: NavierStokes2D.solve_steady 前向路径调用 Rust
- [x] 4.4 验证: Ghia 1982 (Re=100, Re=1000) 结果一致

## Phase 5: 混合伴随求解与 Autograd 融合

- [x] 5.1 自定义 torch.autograd.Function (forward=Rust, backward=vjp)
- [x] 5.2 验证: torch.autograd.gradcheck 通过 (8x4 ~ 32x16 网格, tol=1e-3)
- [x] 5.3 清理: 移除旧的纯 Python 前向路径, 更新 CI

---

## 实施记录

每个 Phase 完成后追加 git log 摘要。
