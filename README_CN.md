# MOVER 与 VitalDB 低血压连续性研究可复现代码包

本包包含分析代码、冻结的分析合同、聚合结果和论文图形，不包含原始数据、病例级派生数据、账号信息或本机绝对路径。

## 先做无数据测试

安装 Python 3.12 与 `requirements.txt` 中的依赖后运行：

```text
python run_smoke_tests.py
```

该步骤使用合成数据，不需要 MOVER 或 VitalDB 原始数据。

## 完整复现

1. 使用 `code/mover_reexport_v5_1.py --root <MOVER根目录>` 生成 MOVER 导出ZIP。
2. 复制并填写 `config.example.yaml`，不要把本机路径、账号或数据上传至公开仓库。
3. 运行 `python run_pipeline.py --config <配置文件>`。
4. 将新生成的聚合结果与 `expected_results/` 比较。

病例级中间CSV不得随论文附件公开。公开仓库前还需要作者确定代码许可证、作者信息和永久存档地址。

## v1.1 相对 v1.0 的变化

- 新增 `code/standardized_risk_intervals_v4_1.py`，给出主对比中两个标准化风险的 95% 区间。
- `code/make_figures_v4_1.py` 新增 `--risk-intervals` 参数，图 3 现在真正画出了图注所述的 95% 置信区间。
- 新增 `code/exposure_support_audit_v4_1.py`：说明预设对比所涉及的发作模式实际有多少病例支持（单次 20 分钟发作仅 6 例，四次 5 分钟发作 0 例）。
- `code/descriptive_observation_audit_v4.py` 现在同时输出 `TABLE1_DESCRIPTIVE_OBSERVED_COHORT.csv`，与论文表 1 的人群一致（此前表格对应的是 parent 队列 3,138 例，与论文表 1 的 2,958 例不一致）。
- `verify_release.py` 增加了对新结果的核对。
