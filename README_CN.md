# 术中低血压与急性肾损伤：血压采样分辨率研究（MOVER / VitalDB / INSPIRE）

本仓库包含该研究的分析代码、冻结的分析合同、聚合结果与论文图形。

## 研究做了什么

**v1.0 预设主分析**：用 MOVER（美国）与 VitalDB（韩国）两个数据库，检验在总时长与平均深度相同的前提下，把 20 分钟术中低血压集中在**一次连续发作**，是否比分散成**四次 5 分钟发作**带来更高的 48 小时肌酐定义急性肾损伤风险。

**v2.0 新增分析**：换了一个问题——**累计时长的关联本身，是否取决于血压"多久记一次"**。用三个队列：MOVER、VitalDB（均为波形数据，按 1 分钟分箱）与 INSPIRE（约每 5 分钟记录一次，用零阶保持处理）。另做内部对照：把两个波形队列**降采样到每 5 分钟一次**，病人、协变量、结局全部不动。

⚠️ **v2.0 的全部内容均为 post hoc（事后）/探索性分析**，不在 `config/ANALYSIS_CONTRACT_v4.yaml` 锁定契约之内，只能作为产生假设的证据来读。

## 包含与不包含

**不包含任何原始数据或病例级派生数据，也不允许上传。** MOVER、VitalDB、INSPIRE 各有自己的数据使用协议；完整复现需要在本机获得三者的授权访问。仓库只放代码、锁定契约、聚合模型报告、审计记录与图形。

## 环境

使用 Python 3.12：

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

macOS / Linux 下把 `.venv/Scripts/python` 换成 `.venv/bin/python`。

## 复现 v2.0 分析

脚本假设 v1.0 流水线已生成的合并特征表存在，并能访问 INSPIRE 源文件。

```bash
python code/build_inspire_v1.py          # 构建 INSPIRE 队列（去重 VitalDB 重叠病例）
python code/downsample_to_5min.py        # 把 MOVER/VitalDB 的 1 分钟网格降采样到 5 分钟

# 三套模型阶梯（时长系数：每 10 分钟低于 65 mmHg 的 OR）
python code/model_ladder.py --input COMBINED_FEATURES_V4.csv \
    --low-column total_low_minutes --center --out model_ladder_1min.json
python code/model_ladder.py --input downsampled_5min_features.csv \
    --low-column ds_total --center --out model_ladder_5min.json
python code/model_ladder.py --input INSPIRE_FEATURES_HOLD.csv \
    --low-column total_low_minutes --out model_ladder_inspire.json

python code/make_figure_resolution.py    # 重画 Figure 1（从数据表重算，不是写死的数字）
```

跑完与 `expected_results/` 中的归档结果比对。

## v2.0 归档结果

累计低血压时长，**每 10 分钟低于 65 mmHg 的校正 OR（95% CI）**：

| 队列 | 分辨率 | 校正 OR |
|---|---|---|
| MOVER + VitalDB | 1 分钟波形 | 1.055 (1.019–1.091) |
| MOVER + VitalDB | 降采样到 5 分钟 | 0.959 (0.928–0.990) |
| INSPIRE | ~5 分钟记录 | 0.948 (0.938–0.958) |

发作模式项在三个队列中均无统计学意义。总时长与模式项的相关分别为 0.88（1 分钟）、0.939（降采样）、0.967（INSPIRE）。

## v1.0 归档主结果

观测到结局的分析队列 2,958 例、200 例急性肾损伤。一次 20 分钟发作 vs 四次 5 分钟发作，校正 OR 1.08696（95% CI 0.88313–1.33783），时长-模式项的似然比检验 P=0.44181。`verify_release.py` 会核对这组数值。

## 解释边界

本研究估计的是关联，不能证明"改变发作连续性"或"改变血压记录频率"就能改变肾脏结局。结局为 48 小时内肌酐定义的急性肾损伤，因无尿量数据，并非完整 KDIGO 判定。

## 版本

- **v2.0.0** — 新增 INSPIRE 队列、5 分钟降采样对照、模型阶梯与分辨率图（全部为事后/探索性分析）
- **v1.0.0** — 冻结的双中心主分析与敏感性分析
