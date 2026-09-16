# K-EmoCon V4.2 实现规格

## 1. 唯一实验

- 数据集：K-EmoCon
- 配置：`configs/kemocon_v4_2_full_window_relation_differential.yaml`
- variant：`lightweight_shared_dynamic_relation_differential_full_window`
- 任务：二分类 arousal 与二分类 valence
- quadrant：由 arousal/valence 概率推导，顺序为 `LALV`、`HALV`、`LAHV`、`HAHV`

配置解析必须拒绝其他 model variant 和非 `multimodal` 训练模式。路径必须是仓库根目录下的相对路径。

## 2. 数据窗口与 manifest

每条 manifest record 表示同一参与者、同一会话中的半开 5 秒时间窗。语音目标采样率为 16 kHz，因此完整窗口为 80000 个采样点。生理信号包含配置中声明的 BVP、EDA、TEMP 通道，并按显式单位、原始采样率和目标时间轴处理。

split 以 debate dyad/session 为隔离单位，训练、验证、测试参与者不得交叉。默认配置使用 5-fold label-stratified rotating splits 和确定性 seed；每个参与者恰好作为测试参与者一次。生理归一化统计只能从当前 fold 的训练分区拟合。每个 fold 必须审计 manifest 声明可用性与实际加载后的模态/通道可用性，完整实验必须汇总 fold 均值与标准差。

## 3. Speech availability 与 activity diagnostics

唯一 availability policy 为 `source_presence`：

- source 存在且读取成功：`speech_available=True`
- source 不存在：`speech_available=False`
- source 声明存在但读取失败：抛出带上下文的 `SourceAdapterError`
- 人工 speech modality dropout：运行时允许将 speech 标记为 unavailable

静音不是模态缺失。全零、近静音、0.2 秒发声、短时发声和持续发声都必须保留完整窗口并执行 speech branch。

`speech_activity_mask` 和 `speech_activity_ratio` 仅用于报告。它们不得控制 availability、compact selection、WavLM、emotion pooling、relation differential、声学条件池化、FiLM、fusion、classifier、sample validity 或 loss。

## 4. Mask 语义

所有项目内部 mask 采用 `True=有效`、`False=padding/缺失`。第三方 API 边界可转换极性。

`speech_attention_mask` 只表示真实采样与 padding。WavLM 输出后的 feature attention mask 必须由该 mask 和 WavLM 下采样长度得到，且是 speech emotion representation、relation differential 与声学条件池化的唯一时序有效性依据。修改 padding 内容不得改变有效输出。

生理 mask 分别表达时间点、通道和模态有效性；全 padding、部分时间缺失、整通道缺失和整模态缺失必须安全处理。

## 5. Speech branch

`WavLMEncoder` 接收完整有效 waveform，一条可用记录每次 forward 仅执行一次 WavLM。WavLM 暴露 12 个 Transformer 层隐藏状态，不包含 embedding output。

- H9–H12：固定均值聚合的情感时序表征
- H1–H2：声学/噪声条件
- relation differential：`NoiseConditionedRelationDifferentialDenoiser`
- modulation：有界 FiLM
- 输出：共享 speech embedding，以及 arousal/valence 辅助 logits

当前稳定配置完全冻结 WavLM，并对 H9–H12 使用固定均值聚合。顶部 H12 微调造成 Arousal Low 塌缩，可学习 H9–H12 聚合又未产生可测收益，因此两项均不保留。全零有效 waveform 必须返回 finite embedding，不能因 activity diagnostics 全 false 而失败。

Speech emotion pooling after relation differential is configurable as
`mean_std` or `attentive_stats`. The baseline is `mean_std`. The attentive
variant scores each valid `[B,T,D]` feature frame with
LayerNorm--Linear--Tanh--Linear, applies a mask-safe softmax using only the
WavLM feature attention mask, and returns concatenated weighted mean and
weighted population standard deviation `[B,2D]`. The H1--H2 noise path retains
the original masked mean/std pooling. Activity diagnostics are not model
inputs.

## 6. Physiology branch

The physiology temporal encoder is an explicit S1/P1 ablation. S1 uses the
established independent two-layer `single_scale` Conv1D stem. P1 uses an
independent `multiscale_dilated` stem for each BVP/EDA/TEMP channel: a shared
1-to-8 kernel-3 projection, three kernel-3 branches with dilations 1/2/4,
concatenation, and a 1x1 projection back to 16 features. Every convolutional
stage zeros invalid timesteps. Both modes retain masked mean/std pooling,
channel availability semantics, preprocessing, and training-fold
normalization. No cross-channel or quality-aware computation is introduced.

当前生理路径使用 `LightweightPhysioEmotionClassifier`：独立通道 stem、mask-aware temporal pooling、共享 physiology embedding，以及 arousal/valence 辅助 logits。可变长度计算不得使用会污染 padding 统计的 BatchNorm。

预处理包括显式通道元数据、可选滤波、flatline/outlier mask、对齐重采样、无效值清零、质量统计和训练集 z-score normalization。质量字段不替代模态 availability。

## 7. Routing 与动态融合

`MultimodalBatchScheduler` 根据 source availability 和人工 modality dropout 构造 compact speech/physiology subbatch，各有效分支最多执行一次，并将结果散射回逻辑 batch。

`FullWindowDynamicMultimodalFusion`：

1. speech embedding 投影到 fusion dimension；
2. physiology embedding 投影到 fusion dimension；
3. 拼接投影结果；
4. 小型 MLP 产生两个 modality logits；
5. 对真实可用模态执行 masked softmax；
6. 加权求和得到唯一 `fused_embedding`。

两模态可用时权重和为 1。speech 缺失时权重为 `[0,1]`；physiology 缺失时为 `[1,0]`；双模态缺失时样本无效且输出安全、有限。activity ratio/mask 不作为 gate 输入。

## 8. 分类与损失

`MultimodalEmotionClassifier` 的 arousal 和 valence head 均从同一个 shared classifier trunk 和 `fused_embedding` 预测，不存在 task-specific fusion weights。quadrant 概率由两项二分类概率推导。

目标函数使用 class-weighted cross entropy；当前单变量采样实验将 speech auxiliary loss 保持为 0.3，physiology auxiliary loss 保持为 0.1。只对 label 与 sample validity 都有效的条目计入损失。人工 modality dropout 属于训练时缺失模拟，但不得产生双模态同时被丢弃的样本。

测试必须启用 same-window modality ablation：只在自然状态下 speech 与 physiology 都可用的测试窗口上分别执行 speech-only 和 physiology-only 推理。该诊断不得参与训练、阈值拟合或 checkpoint 选择。

当前五折对照的训练 sampler 采用柔和的 Valence 类别—参与者均衡权重：Low/High 的期望采样质量分别为 0.35/0.65，同一 Valence 类别内每位参与者等权。该对照与双任务有界 sampler 保持相同模型、split 和损失，用于判断采样策略的真实影响。阈值校准仅使用当前 fold 验证集的 pooled binary macro-F1；不得使用测试标签。

checkpoint 选择使用校准后的 participant Arousal 与 participant Valence Macro-F1 均值，分数严格相同时以更低 validation loss 决胜。当前阈值正则化将验证集网格搜索阈值相对 0.5 的偏移量保留 50%；测试标签不得参与 checkpoint 或阈值选择。当前服务器训练 batch size 为 80，学习率保持 `1e-4`。

## 9. 训练、评估与 checkpoint

训练支持 participant-balanced sampling、梯度裁剪、plateau scheduler、early stopping、阈值校准、断点恢复和小样本 overfit diagnostic。评估按参与者汇总，并报告自然模态模式及可选 same-window modality ablation。

checkpoint 保存模型、优化器、objective、训练状态、CPU RNG 和 JSON-compatible metadata。模型 extra state 必须包含唯一 variant、full-window speech policy、relation differential、共享 dynamic fusion 和共享分类策略等 fingerprint；不匹配时严格拒绝加载。

## 10. 安全与验证

- 不联网下载测试权重；WavLM 单元测试使用本地 tiny config。
- 所有测试使用 CPU 合成数据。
- 所有有效输出必须 finite。
- 双模态均缺失时返回明确无效状态或受控错误。
- 数据源读取失败不得静默降级。
- 项目不实现干净语音预测、去噪波形、波形重建、频谱相减或 source separation。
