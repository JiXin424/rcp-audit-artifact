# Round 5 (2026-07-29) 补实验结果汇总 — 对应审稿 Major/Minor 意见

## E4 (Major #6): 官方 vs 论文 chrF/WER 不一致 — 已解决
- 用 byte-identical hypotheses 复算：官方实现在我们的解码上**精确复现**官方 e19 数值
  (chrF 34.58509769785496, WER 85.77470203767781, ROUGE 35.19608041310144, BLEU-4 12.777)。
- 根因 (chrF 48.05 vs 34.585)：论文版 = 句级平均 + 字符 n-gram 1..4 + 含空格；
  官方 = 语料级 + n-gram 1..6 + 去空格。消融（GT cell）：
  corpus/n6/no-ws=34.585; corpus/n4/no-ws=42.848; corpus/n6/ws=38.465; corpus/n4/ws=46.988;
  sentence/n4/ws=48.049(论文值); sentence/n6/no-ws=35.483。
- 根因 (WER 79.26 vs 85.77)：官方 jiwer 3.1.0 小写化+去标点；标点符号（尤其句尾 " ."）在原始
  WER 中计入参考词数分母且几乎总被正确预测。标点剥离+小写后论文实现 = 85.775（精确一致）。
- 官方协议下全部 28 cells 重算（e4_official_protocol_all_cells.json）：
  Original PURE−GT: chrF +13.23 (论文协议 +13.75), WER −13.90 (sign-flipped +13.90), ROUGE-L +17.87。
  6 个 reconstruction：chrF gaps 全负 (−0.38..−1.94)；WER gaps 全正（PURE 更差 +1.5..+3.5)；
  ROUGE-L gaps −0.15..−1.67。反转在官方协议下仍在 original 上存在、在 reconstructions 上消失——
  且比论文协议更干净（原 seed 606 chrF +0.04 小正值消失）。
- 论文动作：metric_sensitivity 与 cross_metric_gap 两表改为官方协议数值；把自定义协议降为附录敏感性。

## E1 (Major #3): 605-query factorial 完整报告 + 协议修正
- **发现协议错误**：task7 原打分把 25fps 原始姿态直接送入模型（模型不内部降采样；
  model.subsample 属性在发布代码中从未被使用）。附录旧 interaction≈0 是 25fps 下的离协议数值。
- 12.5fps 重打分（e1_factorial_full_report.json，7 评估器 × 6 条件 × 605 项）：
  Original BLEU 2×2: seen_high=11.12, seen_low=2.31, unseen_high=7.19, unseen_low=2.08
  - seen 主效应 +2.08 [+1.56,+2.57] Holm p=0.0003
  - similarity 主效应 +6.96 [+5.93,+8.08]
  - interaction +3.70 [+2.63,+4.79] Holm p=0.0003
  Original NLL: seen 主效应 −0.128 [−0.170,−0.086]; interaction −0.124 [−0.212,−0.039] Holm p=0.0045
  6 seeds (seed级): BLEU seen 主效应 +0.51 [+0.25,+0.77]; interaction −0.39 [−0.92,+0.14] 含0;
  NLL seen 主效应 −0.064 [−0.075,−0.053]; interaction −0.004 [−0.010,+0.001]
- 解释：平衡设计（duration+词数+signer+LaBSE 高低匹配）下 original 的 pool-origin 主效应
  只有 +2.08 BLEU，远小于 Jaccard-only 匹配的 path-D (+8.4)。factorial 的估计更可信，
  D 应降级；interaction 显著说明 original 的 seen 优势随相似度增长。
- factorial 代码 require_same_signer=True（speaker 字段 100% 有值）。

## E2 (Major #2): 替代路径 + Shapley 平均 — B/C/D 路径依赖被定量证实
- 新构系统 SEEN-RAND640-MATCHED-v1（train×640×match 缺失 cell；mean|dJ|=0.052，
  15.1% 查询 |dJ|>0.1——640 池限制匹配质量）。
- 622-support cells (original): S1(7060,max)=23.63, S2(640,max)=16.46,
  S3(7060,match)=17.26, S4(640,match)=14.67, U(test)=8.91。
- canonical path: size +7.17, similarity −0.80, origin +8.35 [7.26,9.42]
- Path1 (size→selection→origin): size +7.17, selection +1.79 [+1.14,+2.46], origin +5.76 [+4.73,+6.81]
- Path2 (selection→size→origin): selection +6.37 [+5.10,+7.67], size +2.59 [+1.65,+3.52], origin +5.76
- Shapley 平均: size +4.88 [+4.02,+5.77], selection +4.08 [+3.32,+4.86], origin +5.76 [+4.73,+6.81]
- 结论：similarity 项从 −0.80 到 +6.37 随排序剧烈变化；origin 项 +8.35→+5.76；
  size×selection 交互大。三项必须改称 path-dependent descriptive contrasts，
  单一归因只能用 Shapley 平均并标注局限。

## E3 (Major #2): 多变量 balance + caliper/signer 敏感性
- 622-support SEEN-MATCHED vs UNSEEN donor 协变量 SMD（e3_balance.json）：
  Jaccard −0.021, LaBSE +0.068, duration +0.014, words −0.003, template density −0.075,
  same_signer −0.111 —— 全部 |SMD|<0.12，边际分布平衡良好。
- 但逐对重叠有限：|Δduration|≤1s 仅 57.2%；|Δwords|≤2 69.0%；双 donor 同 signer 仅 4.7%。
- **事实更正**：released records 的 speaker 字段 100% 有值（train 7060/7060, test 641/641,
  dev 515/515）。论文"speaker field is empty for a substantial fraction"的说法错误，必须删除/更正。
- caliper 敏感性（original, D=SEEN-MATCHED−UNSEEN-PURE 同支持）：
  T005(n=597) D=+8.21 [7.16,9.29]; T010 canonical(n=622) D=+8.35; T020(n=636) D=+8.57 [7.53,9.63]。
- signer 严格匹配(n=580): D=+6.97 [5.87,8.07] —— 符号/显著性不变，幅度降 ~1.4 BLEU。
- pose missingness：发布姿态张量无 NaN/全零关节（0.0%），missingness 不可观测，如实报告。

## E5 (Major #5): 语义槽位外部效标
- 规则式德语天气槽位抽取（NUM/TEMP/TIME/PLACE/EVENT 五族，multiset micro-F1）：
  Original: GT ALL=0.354, PURE ALL=0.526, PT=0.045, PTCOMP=0.424 → 槽位效标确认反转存在。
- 6 reconstructions: GT 0.206..0.239 > PURE 0.177..0.234 全部 → 反转消失（与 BLEU 一致）。
- **决定性 pass-through 控制**：donor transcript 本身 vs query reference slot-F1=0.575；
  original 对 donor pose 的解码达到 0.526（= 上限的 91%）→ original 评估器实质上是在
  复读 donor 原文内容；槽位优势完全由 donor 文本与 query 的词汇重叠解释，
  不能作为"评估器更好识别了真实动作"的证据。

## E6 (Major #4): PHOENIX 语料审计 + 人类参考敏感性
- Hamidullah 风格重叠量化（e6_overlap_contribution.json；审稿人引用有误——实际作者为
  Alkain et al. 2026, Front. Artif. Intell. 9:1743223）：
  top5% 重叠（J=1, n=32）: GT 40.26, PURE 78.73, gap +38.47；其余 95%: GT 11.93, PURE 22.20, gap +10.27；
  最低 50%（J<0.5）: GT 3.50, PURE 11.00, gap +7.50。训练相似样本显著放大但不完全解释反转。
- Czehmann et al. 2026 (LREC 2026, pp.80–92) 公开人类 sign-to-text 参考
  (github.com/DFKI-SignLanguage/sacre-bird-phoenix, CC BY-NC-SA 4.0)。换参考后：
  - 原始参考: GT 12.78 / PURE 23.79 / gap +11.01
  - 人类参考(全641): GT 5.67 / PURE 6.79 / gap +1.12
  - 人类参考(高置信461): GT 7.35 / PURE 8.66 / gap +1.32
  - 6 seeds 对人类参考 gaps: −0.15..−0.65 全负
  - 原始参考 vs 人类参考(as hyp): 11.87 BLEU（translationese 分歧，印证 Czehmann）
- 结论：约 90% 的反转随参考替换消失 → 大反转主要是 reference-validity artifact
  叠加评估器熟悉度效应；剩余 +1.1 也不在 reconstructions 复现。

## E7 (Major #7): bootstrap 升级
- 全 622-support 10,000 次重采样（替代 1000 次 + 100-query 验证）：D [7.26,9.42]。
- 真正分维 cluster bootstrap：template-family [7.24,9.49]、seen-donor [7.11,9.70]、
  unseen-donor [7.12,9.72]；pigeonhole 联合 multiway D [7.66,9.07]。
  与 query-index CI [7.26,9.42] 全部一致 → 供体/模板依赖可忽略。
- fuzzy template family（槽位掩码）：589 clusters/622（仍接近退化，PHOENIX 查询模板唯一性高），
  如实报告；旧"paired multiway bootstrap"更名为 query-index paired bootstrap。
- permutation p 改 (b+1)/(B+1)：headline gap p=1e-4（0 个极端），CI [0, 0.00037]。

## E8 (Major #1): 扩展 rescue + 多目标选择
- Pareto 前沿（e8_pareto_selection.json）：6 seeds 轨迹上 NLL/BLEU/WER 三目标
  Pareto 最优 epoch 全部达不到 gate；BLEU-选择最高 0.0972（seed202 ep50，gate 需 ≥0.1238）；
  WER-选择最低 0.8182（gate 需 ≤0.8049）；gate_reachable_by_any_selection=false。
  BLEU-选择倾向更晚 epoch（95/50/100）vs NLL-选择更早（20-35）→ 两目标分歧。
- 扩展 rescue（bs128/bs512/drop0.2/wd0/ls0.1/ep600lr5e4 × seeds 101/202，各 300 ep）：
  全部未过 gate。最高 wd0_seed202 dev BLEU 0.0992（|Δ|=0.035）；bs512_seed101 0.0916。
  wd0 明显过拟合（train 0.20/val 4.73）。
- 扩展 competence-gap 散点：wd0_seed202（dev 0.0992, 最高）GT test=10.73, PURE=10.98, gap +0.25；
  bs512_seed101 gap −0.71。能力最高的 rescue 仍无反转。

## 协议勘误（论文必须修正）
1. tab:protocol_regression（Table 2）"raw frames / final frames"列错误：
   所有 canonical pose 文件本身即为 12.5fps（GT 32,411; TN-PURE 30,711; PTCOMP 32,400;
   UNSEEN 30,027; MATCHED 29,418; RAND640 32,708），评估时未做第二次 [::2]。
   表中的 final-frames 列（15,515 等）是错误陈述；three-path sensitivity 仍有效（GT 验证）。
2. 附录 factorial 旧数值为 25fps 离协议（见 E1），需替换并说明。
3. speaker 字段 100% 有值，"speaker empty"说法删除。

## 新参考文献
- Czehmann, Yazdani, Hamidullah, Nunnari, Avramidis (2026). "A Sacred Bird Called the Phoenix":
  Auditing the most-used Parallel Corpus for German Sign Language Recognition and Translation.
  LREC2026 12th Workshop on Representation and Processing of Sign Languages, pp. 80–92, ELRA.
  ISBN 978-2-493814-82-1. 数据: DFKI-SignLanguage/sacre-bird-phoenix (CC BY-NC-SA 4.0).
- Alkain, Núñez-Marcos, Escolano, Docío-Fernández, Perez-de-Viñaspre, Labaka (2026).
  Critical analysis of datasets for sign language translation. Front. Artif. Intell. 9:1743223.
  doi:10.3389/frai.2026.1743223. （审稿人误写为 "Hamidullah et al."）
- Artiaga, Kamila, Afli, Lynch, Hasanuzzaman (2025). Rethinking Sign Language Translation:
  The Impact of Signer Dependence on Model Evaluation. Findings of EMNLP 2025, 18379–18391.
  doi:10.18653/v1/2025.findings-emnlp.997.
- Jiang et al. WMT 2025 = 已引用的 jiang2025meaningful（同一篇）。
- SLU-2K: arXiv:2606.03788（语义检查基准，related work 引用）。
- Systemic Biases in Sign Language AI Research: openreview.net/forum?id=oHoCUCZLJo（伦理段引用）。
