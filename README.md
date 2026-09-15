# K-EmoCon V4.2 Full-Window Multimodal Emotion Recognition

本仓库只包含 K-EmoCon V4.2。唯一正式配置为：

`configs/kemocon_v4_2_full_window_relation_differential.yaml`

唯一模型 variant 为：

`lightweight_shared_dynamic_relation_differential_full_window`

## 模型语义

每条记录使用参与者自己的 5 秒、16 kHz 音频窗口，以及同一时间窗内的 BVP、EDA、TEMP 生理信号。

语音可用性采用 `source_presence`：音频源存在并成功读取即为可用。完整静音、极短发声和短时发声都是有效语音窗口；`speech_activity_mask` 与 `speech_activity_ratio` 只用于诊断统计，不参与可用性、路由、WavLM mask、池化、FiLM、融合或损失。

语音路径：完整 waveform → WavLM → 固定均值聚合的 H9–H12 情感表征 → H1–H2 声学/噪声条件 → relation differential → bounded FiLM → speech embedding。当前稳定配置完全冻结 WavLM；顶部解冻和可学习层聚合在 fold 0 未产生可靠的整体收益，因此不保留在生产实验配置中。

生理路径：对齐、伪迹处理与训练集归一化 → 当前轻量生理分类器 → physiology embedding。

融合路径：`FullWindowDynamicMultimodalFusion` 投影两个 embedding，用 learned masked softmax 产生两模态权重，并输出一个共享 `fused_embedding`。Arousal 与 valence 均由同一个共享分类 trunk 和该 embedding 预测。只有真实缺失或显式 modality dropout 才会将相应模态权重硬置为 0。

## 环境与本地 WavLM

项目需要 Python 3.11+，依赖见 `pyproject.toml`。训练和测试不会联网下载模型；WavLM 必须预先放在 `models/wavlm-base`。如需从已可访问的 Hugging Face 环境缓存到该目录，可运行：

```bash
python scripts/cache_wavlm.py --help
```

## 数据准备、校验、训练与评估

所有路径均由配置文件给出，并相对于仓库根目录解析：

```bash
python scripts/prepare_kemocon.py --config configs/kemocon_v4_2_full_window_relation_differential.yaml
python scripts/validate_kemocon.py --config configs/kemocon_v4_2_full_window_relation_differential.yaml
python scripts/train_kemocon.py --config configs/kemocon_v4_2_full_window_relation_differential.yaml --all-folds
python scripts/evaluate_kemocon.py --config configs/kemocon_v4_2_full_window_relation_differential.yaml
```

训练前可用 CPU 离线测试检查当前实现：

```bash
python -m pytest -q
```

## 输出

默认 manifest：`artifacts/kemocon/manifest_self.json`

默认实验目录：`runs/kemocon_v4_2_full_window_relation_differential_seed2026/`

每个 fold 写入 `fold_<index>/`，主要文件包括：

- `best.pt`：严格 fingerprint 的最佳 checkpoint
- `normalizer.json`：仅由训练分区拟合的生理归一化状态
- `run_config.yaml`：实际运行配置快照
- `split_summary.json`：参与者/会话互斥 split 与标签统计
- `history.jsonl`：逐 epoch 训练记录
- `test_metrics.json`：测试评估结果
- `training_summary.json` 和 `train.log`：运行摘要与日志

完整重复实验使用 5-fold label-stratified、dyad-safe rotating splits。每个参与者恰好作为测试参与者一次；全部 fold 完成后写入 `cross_fold_summary.json`，其中包含 fold 均值、总体标准差、逐 fold 指标和各 fold 的运行时模态可用性审计。`split_summary.json` 同时保留 manifest 声明数量与实际加载后的 speech/physiology 可用数量、按参与者差异、按 physiology 通道差异及 mismatch sample ID。

训练采样采用 `bounded_multitask_participant_balanced`：Arousal 与 Valence 的 Low/High 目标采样质量均为 0.35/0.65，以任务—类别内参与者均衡作为初始化，再联合校准两个任务的边际质量；单样本最大/最小权重比限制为 20，避免极少数窗口被反复抽取。验证阈值按 pooled binary macro-F1 选择，Low 与 High F1 等权。

当前单变量采样实验保持 speech auxiliary loss 权重为 0.3、physiology auxiliary loss 为 0.1，仅把采样策略改为双任务有界采样。测试启用 same-window modality ablation，在原本 speech 与 physiology 都可用的同一组窗口上分别屏蔽一种模态，用于判断语音独立判别能力，ablation 不参与训练或 checkpoint 选择。

## 关键约束

- mask 统一为 `True=有效`、`False=padding/缺失`。
- WavLM 的 attention mask 只描述真实采样与 padding。
- 两个模态均可用时融合权重和为 1；单模态缺失时唯一有效模态权重为 1；双模态缺失安全判无效且不产生 NaN/Inf。
- 训练默认包含 participant-balanced sampling、人工 modality dropout、加权交叉熵、阈值校准和 dyad-safe split。
- 项目只做情感识别，不输出增强或重建后的语音。
