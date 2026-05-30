# DiffCFD Rust 改造计划

> 基于 `ddfcfd-reform-1.txt` 五阶段方案制定。原则：不向后兼容，始终最优决策。

## 总体架构

- **前向求解 (Forward)**：纯物理计算，不参与 PyTorch 跟踪 → 下沉到 Rust
- **反向传播 (Backward)**：C1 隐式微分，需在收敛点评估残差 → 保留 PyTorch（vjp）
- **数据通道**：PyTorch Tensor → `.detach().numpy()` → PyO3 零拷贝视图 → Rust 计算 → NumPy 数组返回

---

## Phase 1: Rust 工具链与混合项目初始化

- [x] 1.1 安装 maturin
- [x] 1.2 创建 Rust crate 结构 (`src/` + `Cargo.toml`)
- [x] 1.3 配置 pyproject.toml 使用 maturin 构建后端
- [x] 1.4 引入依赖: pyo3, numpy, ndarray, rayon, faer
- [x] 1.5 实现零拷贝桥梁: 基础 PyO3 函数暴露
- [x] 1.6 验证: `pytest tests/unit/` 通过 + Python 可 `import diffcfd._diffcfd_rust`

## Phase 2: 稀疏矩阵组装加速

- [x] 2.1 Rust: 实现 `build_momentum_u` (CSR 格式)
- [x] 2.2 Rust: 实现 `build_momentum_v` (CSR 格式)
- [x] 2.3 Rust: 实现 `build_pressure_system` (CSR 格式)
- [x] 2.4 Python: NavierStokes2D 调用 Rust 版本替换 _solve_u/_solve_v/_build_pressure_system
- [x] 2.5 验证: Rust 计算结果与纯 Python 版本浮点误差 < 1e-7 (70 unit tests pass)

## Phase 3: SDF 几何计算加速

- [x] 3.1 Rust: 实现 `bspline_sdf` 点到多边形距离计算 (rayon 并行)
- [x] 3.2 Rust: 实现射线投射 winding number 内外判定 (rayon 并行)
- [ ] 3.3 Python: BSplineAirfoil.sdf 保留 PyTorch 实现（需要 autograd 梯度流）
- [x] 3.4 验证: SDF gradient test 通过

> **设计决策**: BSplineAirfoil.sdf 保留 PyTorch 原生实现，因为控制点需要 autograd 梯度。
> Rust `bspline_sdf` 作为可选加速路径，适用于不需要梯度的场景。

## Phase 4: 前向 SIMPLE 求解器完全下沉

- [x] 4.1 Rust: 使用 faer 实现稀疏线性求解 (dense LU for small systems)
- [x] 4.2 Rust: 实现完整 SIMPLE 外循环 (动量步 + 压力修正 + 速度修正)
- [x] 4.3 Python: _run_simple 中的动量/压力系统构建已全部走 Rust 路径
- [x] 4.4 验证: Ghia 1982 (Re=100, Re=1000) 结果一致

## Phase 5: 混合伴随求解与 Autograd 融合

- [x] 5.1 自定义 torch.autograd.Function (forward=Rust kernels, backward=vjp)
- [x] 5.2 验证: torch.autograd.gradcheck 通过 (test_poiseuille_gradcheck + test_lid_driven_cavity_gradcheck)
- [x] 5.3 更新 CI: 使用 maturin 构建 + Rust toolchain

---

## 实施记录

| Phase | Commit | 日期 | 状态 |
|-------|--------|------|------|
| Phase 1 | `f883ce4` | 2026-05-29 | 完成 |
| Phase 2+3 | `766eaf7` | 2026-05-29 | 完成 |
| Phase 4+5 | `01e878c` | 2026-05-29 | 完成 |
| CI 修复 | `8d57ff6` | 2026-05-29 | 完成 ✅ CI全绿 |

### 测试结果

- Unit tests: 102/102 passed
- Poiseuille validation: 2/2 passed
- Lid-driven cavity (Re=100, Re=1000): 2/2 passed
- Gradient checks (gradcheck): 2/2 passed
- Backward step: 1/1 passed
- Grid convergence: 2/2 passed
- Aero (forces, shapes, gradient): 3/3 passed
