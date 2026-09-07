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
4. **架构层面的解读**：gyro warp 是一个 ~30 微秒的矩阵运算，把 IMU 变成了伴随放电；光流是一个每帧都要跑的计算机视觉计算，而且必须先于它服务的补偿步骤执行。对于一个以*廉价常开*的惊讶检测为全部意义的设备，IMU 是对的层次——光流是用来对照的质量上限，残差质量差距（11-20%）是 100 倍成本比下诚实的代价。亚像素 gyro shift 和更好的 f_map 标定能缩小这部分差距。

复现：

```bash
PYTHONPATH=. .venv/bin/python scripts/run_flow_eval.py data/dataset
```

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

```bash
# 挑选 + 下载 segment（probe -> select -> fetch，每步有缓存）
.venv/bin/python scripts/fetch_aoe.py probe --n 60
.venv/bin/python scripts/fetch_aoe.py select --k 8
.venv/bin/python scripts/fetch_aoe.py fetch
# 逐个转换为标准序列布局
for d in data/aoe_src/<segment>; do
  PYTHONPATH=. .venv/bin/python scripts/prep_aoe.py "$d" --out data/aoe/val/seq_XXX
done
PYTHONPATH=. .venv/bin/python scripts/extract_featmaps.py data/aoe
PYTHONPATH=. .venv/bin/python scripts/run_flow_eval.py data/aoe --n-train 0 --n-val 9 --robust
PYTHONPATH=. .venv/bin/python scripts/run_arbiter_eval.py data/aoe --n-train 0 --n-val 9 --robust
```

## 诚实的局限性

- **屏幕录制是此方法的死路（实测，非推测）**。对场景做过一次 `screencapture -v` 录制的端到端测试：静态帧很干净（残差 4e-5），但运动下即使*oracle*补偿器（完美知道真实平移）也留下 ~0.066 的残差，且最佳的整数对齐帧对差异为 1.16/255，而原始帧是 0.009/255——约 130 倍的下限。ProMotion 合成器帧混合 / HEVC B 帧平滑破坏了此方法需要的信号（帧间锐利的时间采样）。结论：在真实相机（iPhone / webcam）上验证，它们采样的是锐利瞬间，而不是屏幕录制。
- **仅旋转补偿**：未建模平移/视差（无加速度计融合）；学习到的权重能吸收一部分，但行走的设备会产生比此合成 demo 更大的残差。
- **合成 demo 是有利的**：真实场景有自动对焦呼吸、曝光自适应——在负像吸收它们之前，这些看起来都像"surprise"。这种吸收正是多时间尺度可塑性的用途，但需要真实数据调优。
- **阈值**：featmap/下游路径现在自校准（`AdaptiveTrigger`——轨迹自身平静统计量上的无量纲 6/2.5 门控）。像素空间 demo 路径仍使用手调 `TriggerConfig` 值；且 regime 检查表明自校准买到的是工作 regime 内的精度，不是 regime 不变性。
- ~~下游价值未测量~~ **已完成**（见上节）：cerebellum 片段训练的检测器每存储字节优于均匀采样——也优于把预算全花在事件帧上的 oracle。

## 下一步

1. **iPhone 实拍**（真正的实验）：按 README 录制手持视频 + IMU；标注事件；跑同样的评估。这是当前的 critical path——屏幕录制捷径已被证伪。
2. 如果想要一个桌面版的中间方案，可做 webcam 对着屏幕的变体：相机通道时间上是锐利的，但需要镜子或对着屏幕的外接相机（内置摄像头拍到的是用户自己）。
3. ~~下游训练对比~~ **已完成**（见上节）：cerebellum 片段在 4 倍字节下胜过 uniform，并在事件 recall 上胜过 oracle。
4. 给 action 向量加入加速度计特征（平移预测）。
