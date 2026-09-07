# 小脑式残差触发数据采集

[English README](README.md) | 中文 README

Perks et al.（Nature 2026, doi:10.1038/s41586-026-10690-6）讨论的想法的最小原型：**相机+IMU 一体化设备不应该存储它能预测的东西**。这套 pipeline 学习设备自身运动（伴随放电 = IMU）的感官后果的"负像"（negative image），只记录残差——即意外（unexpected）的部分。

## 映射关系


| 电鱼 ELL（论文）     | 本原型                                                       |
| -------------------- | ------------------------------------------------------------ |
| 电器官放电（EOD）    | 设备自身的运动                                               |
| EOD 指令的伴随放电   | IMU 陀螺仪数据流                                             |
| 颗粒细胞时间基       | 陀螺特征`[1, |gx|, |gy|, |gz|]`                              |
| 负像（可预测的反应） | 逐 cell 线性预测`w_fast + w_slow` + 慢负像 `neg`             |
| 联合性抑制/易化      | 局部反 Hebb 更新，无反向传播                                 |
| 快/慢可塑性配对      | `w_fast`（秒级）+ `w_slow`（长时），`w_fast` 被拉向 `w_slow` |
| 平衡放电率可塑性门控 | 逐 cell 门控：**当下正在"惊讶"的 cell 不学习**             |
| 输出 cell 残差       | 逐 cell z 分数化残差；内部 cell 取最大值 = surprise          |
| 猎物检测             | 触发式片段保存（带迟滞）                                     |

## 当前 demo 结果（合成 60 秒设备，6 个 surprise 事件）

```
method                recall  precision   stored  clips
cerebellum            0.83    1.00        0.093    5
naive_diff            0.00    0.00        0.034    1
uniform(matched)      0.17    0.20        0.083    5
uniform(5%)           0.00    0.00        0.050    3
```

朴素帧差法淹没在自运动里（即使设备旋转很平滑，recall 也是 0）；与 cerebellum 方法**相同存储预算**的均匀采样漏掉了 6 个事件中的 5 个。cerebellum pipeline 仍漏掉一个事件（最早的那个，此时统计量还在收敛）——见"诚实的局限性"。

## 快速开始

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/make_synthetic.py --seconds 60 --out data/synthetic
PYTHONPATH=. .venv/bin/python scripts/run_experiment.py data/synthetic
```

运行单元测试：

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/
```

## 用 iPhone 真实采集

1. **视频**：用相机 app 录制（1080p30 足够）。
2. **IMU**：运行 *Sensor Logger*（或任何能导出 CSV 的 app），陀螺仪 ≥100 Hz。期望的列：`time,gyroscope.x,gyroscope.y,gyroscope.z`。
3. **同步**：在场景开始前，用力摇一次手机。尖峰在视频和陀螺仪模长中都可见；计算 `imu_offset = t_shake_video - t_shake_imu` 并传入 `--imu-offset`。
4. **真值**：标注 `events.json`：`{"events": [[t0, t1], ...], "fps": 30}`，写入所有应该被记录的时间戳。
5. 运行：`PYTHONPATH=. .venv/bin/python scripts/run_experiment.py data/my_rec`

## 首次真实采集（ webcam，桌面场景）

一段 65 秒的桌面录制，外接 webcam（30 fps）+ 手机陀螺仪数据流（用摇手机技巧同步，`imu_offset` 11.342 秒），6 个标注事件：

```
method                recall  precision   stored  clips
cerebellum            0.33    0.50        0.102    4
naive_diff            0.00    0.00        0.042    2
uniform(matched)      0.17    0.17        0.092    6
uniform(5%)           0.17    0.33        0.046    3
```

相对排序在真实数据上成立（同预算下 cerebellum > 基线），但绝对 recall 从 0.83 掉到 0.33：抓到两个靠后的事件，漏掉四个较早的，且有一个早期片段（5.5–8.9 秒）是误触发。60 秒的桌面场景恰好是 README 局限性预言的困难 regime——自动对焦呼吸效应、曝光自适应、细微背景运动——在负像吸收它们之前，这些都会表现为"surprise"。这段录制早于多时间尺度调优，仍使用手调阈值；把它当作可行性验证（真实传感器噪声不会破坏 pipeline）而非性能测量。

复现：

```bash
PYTHONPATH=. .venv/bin/python scripts/run_experiment.py data/webcam_rec
```

## Adapter 实验：小脑作为冻结 backbone 的插件

把映射从信号层扩展到**现代 CV 技术栈**：冻结一个 ImageNet 预训练的 MobileNetV3-small（576 维池化特征），在后面插入一个小脑 adapter，检验 adapter 的残差是否对下游事件头有帮助。

Adapter（`cerebellum_cancel/adapter.py`）：固定随机颗粒扩展 `phi([f_{t-1}, a_t])` -> 线性预测器 `W_slow`（训练集上 ridge 回归）+ `W_fast`（在线，单位模长门控的反 Hebb，向零衰减）；当下令人惊讶的帧不更新任何参数（平衡放电率门控）；输出 = z 分数化残差 `r_t = (f_t - f_hat_t) / sd`。

基准（`data/dataset`）：50 条合成序列，纹理场景 + 平滑自旋转 + IMU；训练用 1x 运动幅度，验证用 3x（regime shift）。事件 = 物体出现/移动。慢预测器 R^2 = 0.971。

```
variant            AP@1    AP@4   AP@16   AP@40   evRec@10%
A_static          0.536   0.871   0.842   0.815        1.00   （原始特征上的头）
B_slow_only       0.223   0.185   0.157   0.162        1.00   （残差上的头，W_slow）
C_cerebellum      0.193   0.179   0.171   0.186        1.00   （残差上的头，W_slow+W_fast）
C_energy             --      --      --   0.325        0.76   （未训练的 sum(z^2) 读出）
```

**诚实的负面结果。**在这个基准上，adapter 在任何配置下（已训练/少样本/未训练读出）都没有打过原始冻结特征。按根本程度排序的原因：

1. **池化深度特征本来就对运动鲁棒。**1-4 秒的运动 regime shift 完全不伤 A——训练出的头能找到对 blob 敏感的维度，运动变化是可学习的背景。取消机制没有移除任何头自己无法忽略的东西。
2. **残差噪声无法被学习消除。**原始特征 = 信号 + 稳定背景（训练出的头免费减掉背景）；残差 = 信号 + 运动预测噪声（*池化*特征上的线性映射无法表达"内容平移了 k 像素"，所以自运动泄漏进残差）。除非预测器接近完美，取消就是用可移除的背景去换不可移除的噪声——在这里这笔交易是亏的。
3. **事件太显著了。**纹理背景上的白色 blob 在特征空间里 trivially 可分；ELL 的优势（surprise 从被抵消的背景中凸显）需要相对于可预测结构更微妙的事件。

*站得住的部分*：B/C 的 AP 在 1x->3x 运动 regime shift 下不变（两个 regime 都是 0.15-0.19），尽管只在 1x 上训练——残差确实具有运动不变性，且 W_fast 现在能稳定学习（一个最初的 LMS 发散 bug：`fast_lr=0.02` 对比稳定界 `2/||phi||^2 ~ 0.016`，通过单位模长归一化颗粒输出修复）。强版本的机制仍然是像素空间的那个（见上面的"当前 demo 结果"），其特征空间形式就是下一节的几何变体。
*（在显著事件数据集上测得；当前默认数据集是下面使用的 open-set 变体。）*

复现：

```bash
PYTHONPATH=. .venv/bin/python scripts/make_dataset.py --n-train 40 --n-val 10 --seconds 20
PYTHONPATH=. .venv/bin/python scripts/extract_features.py data/dataset   # 需要 torch+torchvision
PYTHONPATH=. .venv/bin/python scripts/train_adapter.py data/dataset
```

## 特征图取消（几何式，无训练）

上面的池化特征 adapter 失败是因为*池化*特征上的线性映射无法表达"内容平移了 k 像素"——自运动作为不可学习的噪声泄漏进残差。本模块是空间结构化的修复方案，也是 ELL 负像更接近的类比：

- backbone 特征图 `F_t`（MobileNetV3-small `features[:3]`，24x28x28，冻结，保留空间结构）；
- **预测** `F_hat_t(x) = F_{t-1}(x - dx_t)`，其中 `dx_t` 来自 IMU 陀螺仪（伴随放电），换算成特征图像素——`dx = f_map * tan(radians(gyro) / fps)`（符号经实测验证）；
- **残差** `r_t = F_t - F_hat_t`，用逐 cell 运行统计量做 z 分数化，统计量只在平静帧更新（平衡放电率门控；warmup 帧无条件引导统计量）；
- **读出**：无训练——池化 z 能量，或最强的内部 cell。全程无任何训练。

基准：同样的 50 条序列，但现在是 open-set 版本——训练事件是微弱 blob（alpha 0.3），3x 运动的验证事件是**不同屏幕区域的微弱长条：一种从未被任何训练头见过的形状**。

```
                          val AP   evRec@10%      （25 条验证序列）
gyro_warp + maxcell        0.304       0.64   （无训练 —— 全场最佳）
gyro_warp + energy         0.209       0.49   （无训练）
C cerebellum (pooled)      0.182       0.91   （残差上训练）
B slow-only  (pooled)      0.165       0.91
naive_diff + maxcell       0.171       0.51   （无训练，无 warp）
naive_diff + energy        0.172       0.34
A static head (pooled)     0.119       0.64   （原始特征上训练 —— 崩溃）
```

三个结论，由弱到强：

1. **陀螺 warp 有效。**平静帧残差下降 2.6 倍（0.48 -> 0.18），且每个 gyro_warp 读出都胜过其 naive_diff 对照（在 3x 运动下）。
2. **surprise 表示可迁移；外观表示不可。**A 的在原始特征上训练的头在未见事件类型上崩溃（recall 0.46）；在*残差*上训练的头保持 0.8-0.9 recall——它们学到的是噪声模型，不是事件本身。
3. **一个固定的、未训练的前端 outright 赢得 open-set regime。**gyro_warp + maxcell（0.271 AP）胜过所有已训练头的变体，零训练、零事件类型假设——架构固化：对齐、局部门控、一个平凡的读出，就是全部机制。

注意：绝对 AP 都不高（3x 运动下的微弱长条对所有方法都很难）；验证集从 10 扩到 25 条时 open-set 优势保持（gyro_warp+maxcell 0.271 -> 0.304 AP，排序不变）；"无训练"仍假设陀螺-平移标定（f_map）已知。

复现：

```bash
PYTHONPATH=. .venv/bin/python scripts/make_dataset.py --n-train 40 --n-val 25 \
    --seconds 20 --event-alpha 0.3 --val-event-shape bar
PYTHONPATH=. .venv/bin/python scripts/extract_featmaps.py data/dataset
PYTHONPATH=. .venv/bin/python scripts/run_featmap_eval.py data/dataset
```

## 下游训练对比（headline claim，已实测）

一个只存储 cerebellum 触发片段的设备，训练下游模型是否**每存储字节更好**？训练帧由各方法在 40 条训练序列上按匹配存储预算采样；下游模型与 adapter 实验相同（冻结 MobileNetV3 特征上的 LogisticHead）；评估在完整的 25 条验证序列上进行（帧级 AP、top-10% 预算下的事件 recall）。片段来源是上面的特征图取消前端（gyro warp + maxcell z^2 读出）——已实测像素空间触发器在此数据集上完全不触发（训练事件微弱，alpha 0.3；其 z < 6 持续门控需要显著事件：阈值降到 2.5 时 recall 也仅 0.13）。

触发是**自校准的**（`AdaptiveTrigger`）：原始 z^2 分数先除以其自身平静帧运行统计量，再进入无量纲 6/2.5 迟滞门控，因此无需手调分数阈值。之前手调的工作点（T=300）保留为参照行。

```
method                stored  frames   val AP   evRec@10%
cerebellum (adaptive)  0.034     826    0.298       0.96
uniform(0.5x)          0.061    1457    0.135       0.58
uniform(1x)            0.074    1766    0.131       0.59
uniform(2x)            0.091    2181    0.125       0.58
uniform(4x)            0.121    2894    0.120       0.57
oracle(adaptive)       0.019     449    0.191       0.72
--
cerebellum(T=300)      0.150    3593    0.210       0.92   （手调参照）
```

结论，由弱到强：

1. **cerebellum 片段在 4 倍字节预算下仍胜 uniform。**3.4% 存储（826 帧）时 AP 0.298 / evRec 0.96；uniform 需要 3.5 倍帧数（12%）仍大幅落败（0.120 / 0.57）。
2. **赢在数据，不在预算。**加大 uniform 预算无法缩小差距（超过 0.5x 后反而更差——背景更稀释），所以优势来自事件富集，而非存储量。
3. **cerebellum 片段甚至胜过 oracle**（预算全花在真值事件帧上），两项指标皆胜（evRec 0.96 vs 0.72）。触发片段保留了事件 onset、消失和运动边界上下文——正是判别式头需要的帧——而 oracle 只保留事件中段帧。
4. **自校准胜过手调阈值。**自适应触发器比 T=300 工作点少存 4.3 倍帧，且训练出更好的头（AP 0.298 vs 0.210）——用轨迹自身平静统计量归一化是比任何固定原始阈值更锐的判据。

Regime 检查（验证集，3x 运动，平均存储占比）：adaptive 0.339 vs T300 0.379。两者在 regime shift 下都膨胀——calm 门拒绝把运动爆发吸收进基线（设计使然），所以噪声大 3 倍的设备仍按比例存更多。自校准买到的是工作 regime 内的精度，不是 regime 不变性；对一个持续新运动水平的慢时间尺度吸收需要比 20 秒更长的流。

Head 稳健性（同样片段、更强下游模型）：把 LogisticHead 换成 128 隐层 MLP **并没有**提高验证 AP——反而降低了（cerebellum adaptive：0.298 -> 0.200；一组 5 配置的正则化扫描——隐层/l2/dropout/迭代数——最高 0.247）。验证 regime（3x 运动 + 未见过的长条形状）惩罚容量：MLP 过拟合训练片段中运动特有的相关性，这与 adapter 节为基于外观的头记录的失败相同（"surprise 表示可迁移；外观表示不可"）。MLP 确实改进了的是排序的头部（cerebellum 片段上 evRec@10% 0.96 -> 0.99-1.00）。在所有 head 和配置下，数据价值 claim 不变：cerebellum 片段在所有匹配预算下胜过 uniform，并在事件 recall 上胜过 oracle。在这个基准上，AP 受限于数据，而非模型。

注意：字节预算用帧数近似（每帧码率按构造跨方法相近）；冻结特征提取器假设随设备部署（adapter 一节的前提）；6/2.5 仍是人为选择的常数，只是无量纲；下游头刻意做小——claim 是关于*每字节数据价值*，而非检测器峰值性能（下文实测了更强的 MLP 头：它并不能提高 AP）。

复现：

```bash
PYTHONPATH=. .venv/bin/python scripts/train_downstream.py data/dataset
```

## 光流基线与成本画像

陀螺 warp 的 CV 原生替代方案是稠密光流：用图像内容自身的运动来补偿，不需要 IMU。在本基准上端到端实测（`cerebellum_cancel/flow_cancel.py`，DIS MEDIUM 档；15 条训练 + 5 条验证序列）：

```
                    平静帧残差（mean |err|）    存储占比  evRec
                    训练 (1x)   验证 (3x)      验证(3x) 验证(3x)
naive（无 warp）       0.093       0.221          --      --
gyro warp             0.079       0.129         0.113   0.30
DIS 光流 warp          0.070       0.103         0.192   0.33
```

**更强的光流估计器在这里打不过 DIS。**把 RAFT（Teed & Deng, ECCV 2020,
raft-things 权重）接进同一条 warp 路径（`scripts/run_raft_eval.py`），
回答"flow 上限会不会只是 DIS 太弱造成的假象"：

```
                    平静帧残差（mean |err|）      ms/帧
                    合成验证集   真实高运动（224px, robust）
naive                 0.221          0.080               --
gyro warp             0.129          0.090             0.033
DIS 光流 warp          0.106          0.052         5.9 / 7.4
RAFT 光流 warp         0.105          0.054       134 / 138
```

（合成验证集 5 条；真实高运动 2 条。）RAFT 在合成上与 DIS 打平
（0.1054 vs 0.1056），在真实上略差（0.0544 vs 0.0522），同时慢 ~20 倍。
flow 上限是任务本身的属性，与估计器无关；gyro 与 flow 的残差差距
无法靠换学习式光流模型弥合。

```bash
PYTHONPATH=. .venv/bin/python scripts/run_raft_eval.py data/dataset --split val --n-seq 5
PYTHONPATH=. .venv/bin/python scripts/run_raft_eval.py data/aoe_motion --split val --n-seq 2 --robust
```

本机每帧延迟中位数（320x240 帧，24x28x28 特征图）：

```
阶段                      ms/帧
gyro shift (24x28x28)     0.032
pixel pipeline 核心        0.171   （warp + 网格 + 可塑性 + 触发器）
冻结 backbone 前向         0.827   （MobileNetV3 features[:3]，224x224，MPS）
DIS 光流 (320x240)         3.813
完整 flow-cancel 步        4.456
```

结论：

1. **光流确实买到残差质量**——平静帧残差比 gyro warp 低 11%（1x）/ 20%（3x）（DIS 追踪的是内容的真实位移，包括陀螺近似漏掉的部分）。这是真实存在的、光流基线最大的优势。
2. **代价是整个像素 pipeline 的 26 倍**（4.5 ms vs 0.17 ms 每帧，仅 warp 一步就差 ~120 倍）——30 fps 预算下像素 pipeline 占帧时间 0.5%，flow 步占 13%。且这些计时是在笔记本上；在端上（手机 CPU、无 MPS）DIS 通常还要慢好几倍。
3. **它依赖纹理和稳定性**：快速运动下 DIS 会跟丢（一条 3x 验证序列存了 59% 的流，gyro 只有 13%——光流噪声会被当成 surprise）；无纹理或低光区域无解，而 gyro warp 不受影响。事件区域会污染局部流场，泄漏进残差。
4. **架构层面的解读**：gyro warp 是一个 ~30 微秒的矩阵运算，把 IMU 变成了伴随放电；光流是一个每帧都要跑的计算机视觉计算，而且必须先于它服务的补偿步骤执行。对于一个以*廉价常开*的惊讶检测为全部意义的设备，IMU 是对的层次——光流是用来对照的质量上限（已用 RAFT 确认与估计器无关），残差质量差距（11-20%）是 100 倍成本比下诚实的代价。亚像素 gyro shift 和更好的 f_map 标定能缩小这部分差距。

**跨平台可移植性**（`scripts/run_portable_check.py`，`runs/portable_check.json`）：唯一依赖设备的阶段是冻结 backbone 前向。同一批帧在 CPU vs MPS 下：特征最大相对差 2e-6，下游 gyro 平静残差差 3.7e-9——取消路径在数值上可跨设备复现。延迟：backbone CPU 1.76 ms vs MPS 1.06 ms（仅 1.7 倍），管道在 CPU 上仍远低于 30 fps 的单帧预算。剩余的未测量边界是手机级硬件（无笔记本 MPS、核更慢）。

复现：

```bash
PYTHONPATH=. .venv/bin/python scripts/run_flow_eval.py data/dataset
```

## 分辨率：决定 gyro warp 值不值的旋钮

gyro warp 只能移除特征图*看得见*的运动。同样的旋转产生的内容位移是 `f_map * tan(omega/fps)` 个特征图像素，而 `f_map` 随 backbone 输入尺寸线性缩放。用三个输入尺寸重跑同一基准（同样的 40 训练 + 25 验证序列，MobileNetV3 features[:3]，stride 8；224 px 时 f_map 24.25 是 rig 默认值），把分辨率单独作为变量：

```
输入    特征图    f_map   平静帧残差（验证集）  gyro 收益   gyro shift   DIS 光流
        尺寸             naive   gyro   flow   vs naive  ms/帧        ms/帧
224     28x28    24.25   0.221   0.130  0.107  -41%       0.032        3.86
336     42x42    36.38   0.187   0.089  0.069  -52%       0.061        4.20
448     56x56    48.50   0.169   0.072  0.052  -58%       0.087        4.35
```

（backbone 前向在 MPS 上各尺寸都 ~0.6 ms/帧——浅层 stride-8 是访存瓶颈；分辨率的协议成本在存储和带宽，不在算力。同一会话重测，`runs/flow_eval_res*.json`。）

收益单调且未饱和：每提高 1.5 倍输入尺寸，就多消掉 6-10 个百分点的 naive 预测器动不了的平静残差。与光流的差距也随分辨率收窄（flow/gyro 0.82 → 0.73）：gyro warp 的主要误差源——位移的亚像素量化——随特征图变精细而缩小。这也反向解释了真实数据的两个 regime：桌面级 Open-AoE 在 224 px 下 f_map ~11-19、旋转 < ~5°/s（位移 < 0.05 特征图像素——没东西可消，warp 中性）；而 p95 |yaw| 31-52°/s 的高运动对在 ~0.5-0.7 特征图像素——进入了模型误差（roll 未建模、标定）开始*增加*残差而非消减的窗口。

协议层设计规则：**伴随放电取消是一个被分辨率门控的原语**——配细粒度特征（DINOv2 patch token，或 ≥448 px 输入的 stride-8 特征图），它用 ~0.09 ms/帧消掉一半可预测残差；在粗特征图上它免费但中性。同一个变量也决定*触发器*是否工作：448 px 下门控自归一化触发器在全部 5 条合成验证序列上冻到 0 clips（基线 surprise 随分辨率增长并重尾化），而 winsorized 稳健统计在 5 条上全部触发——真实视频的门失效，在合成数据上复现了。

复现：

```bash
# 其它输入尺寸的 featmaps；f_map 线性缩放（24.25 * size/224）
for S in 336 448; do
  .venv/bin/python scripts/extract_featmaps.py data/dataset_$S --size $S
  PYTHONPATH=. .venv/bin/python scripts/run_flow_eval.py data/dataset_$S \
      --n-train 40 --n-val 25 --size $S
done
```
（`data/dataset_$S` 目录里是帧的符号链接、`imu.csv` 副本和缩放过 `f_map` 的 `events.json`；可随时删除重建——featmaps 几分钟就能重提。）

## 光流仲裁级联

介于"只用 gyro"和"每帧跑光流"之间的中间架构：gyro warp 常开；DIS 光流**只在触发候选帧上**被调用，作为仲裁者否决它能解释的候选。每个候选 run 峰值帧的组成方式（`scripts/run_arbiter_eval.py`）：gyro shift 同时预校正灰度帧对（这样 DIS 只需搜索 gyro 模型预测漏掉的小残差——不做预校正时 DIS 在大旋转下跟丢，而触发恰恰在那里开火）和 F_{t-1}；然后 DIS 沿残差流精化预测。当光流精化后的残差内部能量 < gyro 残差的 `explain_ratio`（0.5）时否决该 run——即该残差是自运动建模误差，不是 surprise。

全基准实测（15 训练 + 5 验证，run 级读数——固定阈值 run，与 `AdaptiveTrigger` 的自适应 clip 不同，它对 trace 是单调的）：

```
                 stored   precision   evRec   DIS 调用/序列   摊销 ms/帧
train gyro        0.401      0.257      1.00        --            --
train cascade     0.401      0.257      1.00       21.3         0.135
val   gyro        0.520      0.146      1.00        --            --
val   cascade     0.498      0.155      1.00       23.6         0.150
```

诚实解读：在此合成基准上级联几乎是一个 no-op（118 个验证 run 只否决 6 个）。这是基准的*实测*属性，不是仲裁器的弱点——三条机制解释，每条都验证过：

1. 合成基准是纯全局 yaw/pitch 平移（roll < 1 度/秒），恰好在 gyro warp 的模型类内，所以 warp 近精确，平静帧残差是特征噪声而非建模误差（光流精化后的残差能量 ≈ gyro 残差能量，比值 1.03–1.27）。
2. 在 28×28 特征图上，亚像素标定误差仍在特征噪声下限之下：把 f_map 拉偏 10–50% 后 `rho` 仍 ≈ 1，不会触发否决（评估脚本的 `--fmap-scale`）。
3. 仲裁器*本来能*抓到的东西——全局平移模型无法表达的结构化残差（真实 roll、视差/深度、大标定误差）——在这份数据里不存在。

所以级联的当前定位是：机制与成本已验证（摊销 ~0.15 ms/帧，比每帧跑 DIS 便宜 ~25 倍，需要时保留否决权），价值待真实相机验证——它恰好是为合成基准所缺的那些失效模式准备的安全网。run 级读数还暴露了 `AdaptiveTrigger` 的_running 归一化不是单调的_（清零一个被否决的 run 可能让后续 clip *更敏感*），级联评估通过在固定阈值 run 级对比回避了这一点。

```bash
PYTHONPATH=. .venv/bin/python scripts/run_arbiter_eval.py data/dataset
```

## 真实第一人称验证（Open-AoE）

合成 rig 是纯全局 yaw/pitch，所以第一次真实世界检验使用 [Open-AoE](https://github.com/ant-research/Open-AoE)（HF 上的 inclusionAI/OpenAoE-2000h）：~700 小时消费级手机第一视角视频，带视觉 SLAM 相机轨迹和时间对齐的原子动作标注。从 60 段探测样本中用 `scripts/fetch_aoe.py` 按场景/动作多样性贪心挑选了 9 个 segment（浴室/卧室/儿童房/餐厅/厨房/客厅/工坊/工作台/商店，各 134–301 秒，8 段共 62.0k 帧），用 `scripts/prep_aoe.py` 准备：对 `cam_c2w` 差分得到 pipeline 期望的 gyro 式信号（逐轴符号对照 DIS 流自标定，|r| = 0.18–0.93，最差的两段是几乎静止的画面，没有运动可供相关），写出标准序列布局，并使用每部手机真实焦距（`f_map` = 10.95–19.17，fx 751–1315，而非合成 rig 的 24.25）。

先在一个 segment（seq_000）上实测，再推广到全部九个——三条合成 rig 给不出的发现：

1. **真实自运动在 28×28 上是亚像素的**。内容位移中位数 0.013 特征图像素（p95 0.05，max 0.16），所以 gyro warp 在物理上是中性的：平静残差 naive 0.0323 vs gyro 0.0328。伴随放电 warp 只有在更细分辨率、更深层特征、或行走级运动下才物有所值——桌面第一视角 + 224 px 输入的场景里用不上。
2. **平衡门在真实数据上会冻结**。96.4% 的帧 surprise > 4，warmup 之后运行统计只在 3.6% 的帧上更新，sd 单向衰减（0.92 → 0.17）；z 饱和（max|z| 中位数 = 13），两种读数朝相反方向退化：固定阈值 run 覆盖全部帧的 92%，而 `AdaptiveTrigger` 的相对归一化**一次 clip 都不触发**。平静门假设（surprise 是稀疏的）对 pop-in 式合成事件成立，对真实视频不成立。修法是 winsorized always-update 统计（`CancelConfig.robust`）：离群值以 3 倍 sigma 截断后进入运行统计，既撑不大尺度也不会把门冻住。换成稳健统计后触发器恢复工作：**stored 9.9%、事件 recall 0.79（此前为 0 clips）**。一个有力的旁证：DIS 光流基线（`FlowWarpCancel`）共用同一套门控统计，在真实视频上同样 0 clips——失效的是门，不是预测器。
3. **残差尖峰是真实的，但与动作标注不对齐**。残差最高的帧是快速手/物体运动+运动模糊（`runs/aoe_peaks/`），但分数与标注的动作区间*反*相关（AUC 0.47）：原子动作是运动能量起伏的长区间，而残差索引的是运动瞬变。因此帧级 event-recall 对照人类动作标注在真实数据上是错误指标；对残差触发*存储选择*架构来说，有意义的指标是被选片段的下游价值（见上节），而不是标注对齐。

三条发现在全部九个 segment 上都成立：gyro-warp 的平静残差与 naive 的差别从未超出噪声（各段自运动都是亚像素，内容位移中位数 < 0.05 特征图像素）；稳健统计触发器在每段都工作（stored 1.2–13.1%，中位 9.2%；对动作标注的事件 recall 0.14–0.79，中位 0.46——seq_000 上的离散结论，现在带上了误差范围）；DIS 光流基线在同一套门控统计下依然大多 0 clips，除了光流自己产生幻觉触发的一段（seq_004：stored 99%）。每序列数字见 `runs/flow_eval.json`。

run 级别的光流仲裁同样推广（`runs/arbiter_eval.json`）：九个片段上稳健统计产生约 140 个候选 run/段，级联否决其中的 8–31%（光流调用摊销到 0.89 ms/帧），把被标记帧的存储量从 0.89 降到 0.72，precision 保持 0.994，事件 recall 1.00 → 0.94；clip 级别上 AdaptiveTrigger stored 7.9% → 20.6%、recall 0.49 → 0.55——被否决的 run 即便在亚像素位移下也主要由自运动建模误差主导。

**高运动端**（259 段探测池里运动 top 的 2 段，p95 |yaw| 31–52°/s，桌面片段为 1–41；`data/aoe_motion`、`runs/flow_eval_motion.json`）终于离开了亚像素 regime——预测器的排序随之翻转，但带保留。平静残差：gyro warp 现在比 naive *更差*（0.090 vs 0.080；二维平移模型不补偿 roll，且该段符号标定更弱，仓库段 y r = 0.15），而 DIS 光流在质量上明确胜出（0.052）——成本上则以 ~250 倍落败（8.4 ms vs 0.033 ms/帧）。稳健触发器也变稀（stored 1.4%/1.2%，evRec 0.09/0.04）：连续真实运动让残差重尾化，winsorized sd 膨胀，z 阈值水涨船高。仲裁正是这个 regime 的桥：在最活跃的片段上否决 11% 的 run（stored 0.92 → 0.62，precision 1.00，摊销 0.58 ms/帧）。诚实的底线：在这个语料上即使 top-decile 的旋转量在 28×28 上也只有 ~0.5 特征图像素/帧，所以 gyro warp 在该分辨率下的质量窗口很窄——它的理由是成本，光流仲裁是廉价模型失效时的付费升级。（n=2，且上述标定保留成立。）

**448 px 分辨率交叉验证**（同一对片段，特征图翻倍至 38×38；`data/aoe_motion_448`、`runs/flow_eval_motion448.json`）：设计规则"特征越细 gyro warp 越值"——在合成数据上单调成立（224/336/448 收益 -41%/-52%/-58%）——**不能迁移到真实高运动**。gyro 依然比 naive 差（0.0535 → 0.0614，+15%），DIS 光流逐段翻转（走廊段 0.086、仓库段 0.051；均值 0.068，两段中有一段比 naive 还差）。在真实数据上，损伤主要由纯旋转模型看不见的东西主导——平移/视差、滚动快门畸变——而更多像素让它们*更显眼*而非更不显眼。分辨率修的是估计器误差，修不了模型类误差。下一个要补的模型缺口是平移，不是分辨率——而补平移比看起来难：

**平移项：实测，负收益**（`scripts/run_trans_eval.py`、`runs/trans_eval_data_aoe_motion_val.json`）。SLAM 轨迹同样能差分出 cam 系平移速度，`prep_aoe.py` 会逐段标定一个常深度比例（1/Z，对 DIS 流最小二乘——与 gyro 符号标定、f_map 同级）。在高运动对上，旋转+平移 warp 比纯旋转*更差*（calm residual 0.0901 → 0.0938，两段皆然）。均匀平移解释了均值流却造成逐格错位：真实场景深度结构化（走廊近墙和远端移动不同），常深度模型类是错的，任何逐段标量标定都修不了。平移补偿要么带深度结构（单目深度、平面模型），要么留在按需光流仲裁那条路上。

```bash
# re-prep 会写出平移列 + 逐段 trans_scale
PYTHONPATH=. .venv/bin/python scripts/prep_aoe.py data/aoe_src/<segment> \
    --out data/aoe_motion/val/seq_XXX
PYTHONPATH=. .venv/bin/python scripts/run_trans_eval.py data/aoe_motion \
    --split val --n-seq 2
```

```bash
# 挑选 + 下载 segment（probe -> select -> fetch，每步有缓存）
.venv/bin/python scripts/fetch_aoe.py probe --n 60
.venv/bin/python scripts/fetch_aoe.py select --k 8
# 高运动 regime：先按轨迹旋转量过滤再选
.venv/bin/python scripts/fetch_aoe.py select --k 3 --motion-min 60
.venv/bin/python scripts/fetch_aoe.py fetch
# 逐个转换为标准序列布局
for d in data/aoe_src/<segment>; do
  PYTHONPATH=. .venv/bin/python scripts/prep_aoe.py "$d" --out data/aoe/val/seq_XXX
done
PYTHONPATH=. .venv/bin/python scripts/extract_featmaps.py data/aoe
PYTHONPATH=. .venv/bin/python scripts/run_flow_eval.py data/aoe --n-train 0 --n-val 9 --robust
PYTHONPATH=. .venv/bin/python scripts/run_arbiter_eval.py data/aoe --n-train 0 --n-val 9 --robust
```

## 真实数据下游价值（负结果）

合成的 headline 结论（cerebellum 片段每存储字节训练价值更高）**不迁移**到真实 egocentric 视频——已实测，带保留。

任务层面被数据强制改了一处："主动操作"标注在 9 段中的 8 段覆盖 ~100% 帧，binary 事件检测任务退化。真实数据任务改为帧级动词分类：取原子动作标注中 7 个目标动词（fold、grasp、insert、set_down、thread、unfold、wipe，每类在训练/验证两半各 ≥300 帧；`scripts/make_verb_labels.py`），one-vs-rest logistic 头 + 冻结 576 维 pooled 特征，6 段训练 / 3 段验证（`scripts/train_downstream_aoe.py`、`runs/downstream_aoe_results.json`）：

```
方法              存储占比  帧数    val mAP  区间recall@10%
cerebellum        0.071    3296    0.197      0.75
uniform(0.5x)     0.035    1643    0.209      0.68
uniform(1x)       0.071    3317    0.192      0.75
uniform(2x)       0.143    6696    0.204      0.77
uniform(4x)       0.290   13516    0.193      0.73
oracle            0.056    2528    0.203      0.80
```

触发器本身工作正常（stored 7.1%、区间 recall 0.75），但选中的帧不是判别帧：残差触发存的是*运动学上不可预测*的帧（运动突变），而动词身份是语义信息、分布在整个操作区间——与残差峰对动作标注帧级反相关的实测（AUC 0.47，见上）一致。mAP 散布（0.19–0.21）在噪声量级；单 split、无 seed 误差棒。适用范围结论：价值主张在它设计的稀疏物理事件 regime（跌落、抓取失败、异常）成立；稠密语义任务的采集筛选需要别的信号。

```bash
PYTHONPATH=. .venv/bin/python scripts/extract_features_aoe.py data/aoe
PYTHONPATH=. .venv/bin/python scripts/make_verb_labels.py data/aoe
PYTHONPATH=. .venv/bin/python scripts/train_downstream_aoe.py data/aoe
```

## 诚实的局限性

- **屏幕录制是此方法的死路（实测，非推测）**。对场景做过一次 `screencapture -v` 录制的端到端测试：静态帧很干净（残差 4e-5），但运动下即使*oracle*补偿器（完美知道真实平移）也留下 ~0.066 的残差，且最佳的整数对齐帧对差异为 1.16/255，而原始帧是 0.009/255——约 130 倍的下限。ProMotion 合成器帧混合 / HEVC B 帧平滑破坏了此方法需要的信号（帧间锐利的时间采样）。结论：在真实相机（iPhone / webcam）上验证，它们采样的是锐利瞬间，而不是屏幕录制。
- **仅旋转补偿**：未建模平移/视差（无加速度计融合）；学习到的权重能吸收一部分，但行走的设备会产生比此合成 demo 更大的残差。常深度平移项已实测为*负收益*（见上节）：真实场景深度结构化，平移需要带深度结构的模型。
- **合成 demo 是有利的**：真实场景有自动对焦呼吸、曝光自适应——在负像吸收它们之前，这些看起来都像"surprise"。这种吸收正是多时间尺度可塑性的用途，但需要真实数据调优。
- **阈值**：featmap/下游路径现在自校准（`AdaptiveTrigger`——轨迹自身平静统计量上的无量纲 6/2.5 门控）。像素空间 demo 路径仍使用手调 `TriggerConfig` 值；且 regime 检查表明自校准买到的是工作 regime 内的精度，不是 regime 不变性。
- ~~下游价值未测量~~ **合成上已完成**（见上节）：cerebellum 片段训练的检测器每存储字节优于均匀采样——也优于把预算全花在事件帧上的 oracle。**真实 egocentric 视频上优势未迁移**（见上节）：触发器选的是运动不可预测的帧，不是语义重要的帧。
- **真实数据下游表是单 split、单 seed**：mAP 散布在噪声量级；下一个是 3-fold 跨段 + 误差棒重跑，再下结论。

## 下一步

1. **iPhone 实拍**（真正的实验）：按 README 录制手持视频 + IMU；标注事件；跑同样的评估。这是当前的 critical path——屏幕录制捷径已被证伪。
2. 如果想要一个桌面版的中间方案，可做 webcam 对着屏幕的变体：相机通道时间上是锐利的，但需要镜子或对着屏幕的外接相机（内置摄像头拍到的是用户自己）。
3. ~~下游训练对比~~ **合成上已完成**（见上节）：cerebellum 片段在 4 倍字节下胜过 uniform，并在事件 recall 上胜过 oracle。真实数据重跑：优势未迁移（见上节）——排队中的后续：3-fold 跨段 + seed 误差棒，然后在触发器本该赢的稀疏物理事件 regime 做正向验证。
4. ~~给 action 向量加入加速度计特征（平移预测）~~ **已实测，负收益**（见上节）：常深度平移 warp 帮倒忙；需要深度结构（单目深度/平面模型），或者留在按需光流仲裁。
