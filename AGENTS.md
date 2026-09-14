# K-EmoCon V4.2 工程规则

本文档适用于仓库内全部代码、配置、测试和文档。

## 唯一产品范围

- 唯一实验配置：`configs/kemocon_v4_2_full_window_relation_differential.yaml`
- 唯一 model variant：`lightweight_shared_dynamic_relation_differential_full_window`
- 唯一训练模式：speech 与 physiology 的 multimodal 训练
- 主输出：`arousal_logits: [B,2]`、`valence_logits: [B,2]`
- quadrant 只由两项概率推导，顺序固定为 `LALV`、`HALV`、`LAHV`、`HAHV`

不得增加历史实验兼容分支、替代数据集入口或独立单模态模型入口，除非用户明确变更项目范围。

## 工作区与版本控制

- 所有成果只保存在当前本地项目目录。
- 禁止执行任何 Git 命令，禁止连接或使用 GitHub。
- 不下载或伪造真实数据。
- 不擅自安装或修改全局 Python 环境；缺少依赖时准确报告。
- 不隐藏假设、未完成项或失败检查。
- 修改必须保持当前数据 split、label、preprocessing、训练和评估契约，除非用户明确要求改变。

## Full-window speech 语义

- 每个有效参与者音频窗口为 5 秒、16 kHz。
- availability policy 只能是 `source_presence`。
- source 存在且成功读取即 `speech_available=True`，包括全零和任意短时发声。
- source 不存在才自然 unavailable；读取失败必须抛出 `SourceAdapterError`。
- 显式 modality dropout 可以人工将 speech 标记为 unavailable。
- speech activity mask/ratio 只用于诊断，不得影响模型计算、路由、融合或损失。

## Mask 约定

- 项目内部统一 `True=有效`、`False=padding/缺失`。
- `speech_attention_mask` 只表示真实采样与 padding。
- WavLM feature mask 只能由 attention mask 和 WavLM 下采样规则得到。
- padding 内容变化不得影响有效池化或分类结果。
- 所有时序模块必须安全处理可变长度、单点有效、全 padding、通道缺失和模态缺失，不得产生 NaN/Inf。

## 固定模型结构

Speech：WavLM → H9–H12 emotion representation + H1–H2 acoustic/noise condition → relation differential → bounded FiLM → speech embedding。

Physiology：显式通道预处理与训练集归一化 → `LightweightPhysioEmotionClassifier` → physiology embedding。

Multimodal：availability-safe routing → `FullWindowDynamicMultimodalFusion` → learned masked-softmax weights → 一个 shared `fused_embedding` → shared classifier trunk → arousal/valence heads。

融合不得使用固定平均。单模态缺失时其权重为 0、唯一有效模态权重为 1；双模态缺失必须明确无效且数值安全。activity diagnostics 不得进入 gate。

## 训练与 checkpoint

- 默认损失为 class-weighted cross entropy。
- speech/physiology auxiliary loss 权重由唯一配置给出。
- 保留 participant-balanced sampling、dyad-safe split、modality dropout、阈值校准、overfit diagnostic 和 participant-independent evaluation。
- checkpoint 必须使用当前 V4.2 严格 fingerprint，配置或计算语义不一致时拒绝加载。

## 禁止语音增强

项目只做情感识别。禁止实现干净语音监督、去噪波形、波形重建、频谱相减、source separation 或任何语音增强业务接口。

## 实现与清理要求

- 优先使用 `rg` 搜索文件和文本。
- 文件编辑使用 `apply_patch`。
- 不删除或覆盖与任务无关的用户数据和运行结果。
- 只保留当前实现真实引用的 production symbol 和 export；删除代码时同步清理 imports、typing、tests 与文档。
- 公开类和函数 docstring 必须说明输入与输出张量 shape。

## 测试与完成报告

- 测试不得联网或下载预训练权重。
- WavLM 测试使用本地构造的 tiny `WavLMConfig`。
- 测试只使用 CPU 和小型合成张量。
- 阶段结束前运行实际可用的 pytest、Ruff 和 mypy；未安装工具不得声称通过。
- 测试失败时不得声称完成。
- 报告必须列出修改/删除文件、关键选择、实际命令与真实结果、尚存问题和明确未实现内容。
