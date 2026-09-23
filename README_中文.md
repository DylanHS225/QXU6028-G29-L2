# G29 L2 完整计算程序

## 最简单的运行方式

将 `reproduce.py` 保存到 Downloads（不要放在待生成的“L2提交”文件夹中）。
安装依赖后运行：

```bash
python3 -m pip install "numpy>=2.0,<3" "pandas>=2.0,<3" "matplotlib>=3.7,<4"
python3 /Users/dhs225/Downloads/reproduce.py
```

默认将全部结果保存到 `/Users/dhs225/Downloads/L2提交`。
脚本已内嵌原始学生包的 **G29** L2 CSV、元数据、模板和校验清单，
不需要再次下载或手动填写数据。内嵌部分采用 Base64 只是为了逐字节保留原文件，
没有隐藏答案或硬编码拟合结果；所有结果仍由积分、拟合和重采样计算得出。
Python 需为 3.10 或更高版本。生成真实提交哈希需安装 Git。

如果目标目录已含文件，程序停止，不会覆盖。请先重命名旧目录备份，或运行：

```bash
python3 reproduce.py --output "/Users/dhs225/Downloads/L2提交_第二次"
```

## 输出文件

| 文件 | 作用 |
|---|---|
| `G29_L2_Checkpoint.zip` | 本次 L2 checkpoint 上传包，核心文件直接放在 ZIP 根目录 |
| `results.json` | 完整模板结构，填入 L2 的四个字段与真实 pipeline_commit |
| `02_model_fit.png`、`03_EF_trajectory.png` | 两张必需主图 |
| `brief_note.md` | 英文方法、残差对比、稳定性、分辨率限制和器件建议 |
| `commit_hash.txt` | 产生结果的真实本地 Git commit |
| `01_raw_repeats_zero_crossing.png` | 三组原始重复、均值/标准差、Hall 零交叉 |
| `04_gap_selection_bootstrap.png` | 带隙搜索、15% 判据和重采样分布 |
| `05_residuals_cone_wing_gate.png` | 残差及 cone 两翼的平方关系检查 |
| 同名 `.svg` | 可编辑矢量图，数值与 PNG 相同 |
| `L2_diagnostics.json` | bootstrap 完整区间、稳定性比例、网格敏感性、原始文件哈希等 |
| `EF_trajectory.csv` | 全部 58 个 EF、点位区间、最近网格点与残差 |
| `processed_pairs.csv`、`gap_scan.csv` | 均值/标准差/归一化数据及全部候选残差 |
| `bootstrap_replicates.csv`、`bootstrap_arrays.npz` | 200 次重采样的结果及每次抽中的配对行 |
| `model_library.npz` | 规定网格上的理论曲线和 populations |
| `run_config.json`、`run_log.txt`、`validation.json` | 参数、运行摘要与自动检查记录 |
| `reproducible_project` | 代码、原始数据、测试和真实本地 Git 历史 |
| `L2_source_history.bundle` | 可携带的真实 Git 历史，可用于干净克隆验证 |
| `AI_usage_log.md`、`AI_error_ledger.md` | 本次已知 AI 使用与问题记录，需补充你们组的实际使用情况 |

## 严格执行的课堂方法

1. 按 measurement_index 汇总三次重复，保留原测量顺序与样本标准差。
2. Hall 保留符号，以最大绝对值归一化；片电阻独立以最大值归一化。
3. E 使用 [-2.5,2.5] eV 的 4,000 点网格。按指南用 `np.where` 掩码后在完整网格上做梯形积分。
4. EF 使用 [-8,+9] kBT 的 400 个候选点。比较 cone 与 31 个 hard-gap 候选。
5. 使用二维欧氏最近离散点匹配，RMS 为每个点平方距离均值的平方根。
6. 只有全局最佳 hard-gap 残差严格小于 0.85 倍 cone 残差且 Eg >= 1 kBT，才选 hard gap。
7. 200 次 bootstrap；每个 index 抽取一个完整的配对重复行，重新归一化并重新选模型。
8. 主结果来自重复均值。bootstrap 中位数不替代主结果，EF 保持测量顺序。

采用指南常数 kB=8.617333e-5 eV/K，NA=3.816e15 cm^-2。
固定正的 c、e、mu 只影响尺度，归一化前取 c=e=mu=1。
记录固定随机种子 20260923；可用 `--seed` 修改并记录。

## 不确定度和数据解释

模板的 `Eg_uncertainty_kT` 没有进一步规定类型，此实现采用标量 `(P84-P16)/2`。
完整 P16/中位数/P84 单独存储，不把半宽冒充 95% 置信区间。
本组全部 bootstrap 均选择 cone，因此该字段可为零；这不证明真实带隙严格为零，
也不证明误差为零。正确表述是“本数据与规定方法未分辨出硬带隙”。
EF 区间是逐点的重复重采样区间，不是整条轨迹的同时置信带。
模型曲线的小锯齿来自规定的有限积分网格与 EF 掩码，不要为了美观擅自平滑后重新拟合。
额外的 8,000 积分点和 800 EF 点仅用于诊断，不会替换规定网格的提交结果。

## Git 与课程提交边界

脚本只在新建的 `reproducible_project` 中自动创建一次真实本地 Git 提交，
署名为 `L2 generated snapshot`。不会修改你的其他仓库，不会连接或上传 GitHub/Gitee，
不会伪造小组成员分工或历史。该提交包含可复现代码和原始数据。

课程要求的教师可访问 GitHub/Gitee 仓库、真实分工历史、成员签名，以及最终整合
L1/L2/L3 的 `reproduce.py`，仍需你们组实际完成。当前脚本与上传包只负责 L2。
`results.json` 保留 L1/L3 模板字段而不虚构内容，因此不能当最终课程全阶段结果提交。

Git 不可用时会完成数值和图表，但明确不生成正式 checkpoint ZIP。
不要把占位文字当哈希。macOS 上可先在终端检查 `git --version`，如系统提示安装
Command Line Tools，按系统流程安装，然后使用新输出目录重跑。

## 干净克隆复现与测试

```bash
git clone "/Users/dhs225/Downloads/L2提交/L2_source_history.bundle" L2_clean
cd L2_clean
python3 -m unittest -v test_l2.py
python3 reproduce.py --data data/regular_G29 --output L2_output --checkpoint
```

高级模式可使用你们自己的未修改数据文件夹：

```bash
python3 reproduce.py --data data/regular_G29 --output L2_output --group G29 --checkpoint
```

如果希望保留已有 L1/L3 字段，可加 `--base-results existing_results.json`；
该文件也需要属于已提交版本。它只是保留已有字段，不会重新计算 L1/L3。

在 Spyder/PyCharm 中可直接运行本文件；在 Jupyter 中请保存成 .py 后运行
`!python3 /Users/dhs225/Downloads/reproduce.py`，不要把整个脚本粘贴进带额外参数的内核。

依据：Practical Methods Guide 第 3、6–9、14 页，Assessment Brief 第 5.2 节。
隐藏真实参数和评分器未提供；测试证明实现与规定方法一致，不构成分数保证。
