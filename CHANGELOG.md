# Changelog

## 2026-05-29

- Created a clean remote-sensing solid-waste baseline project.
- Added configurable binary classification training for 4-channel `.npy` patches.
- Added baseline models: `small_cnn`, `resnet18`, `resnet50`, `mobilenet_v3_small`, `efficientnet_b0`, and `convnext_tiny`.
- Added evaluation with confusion matrix metrics, Kappa, ROC-AUC, PR-AUC and threshold sweeps.
- Added large-area patch inference that exports CSV predictions.
- Added data audit tooling.
- Added `data_v2` conversion tooling to split large `.npy` arrays into inspectable single-sample `.npy` files, preview PNGs and `samples.csv`.
- Converted the local default `4201` dataset into `data_v2` locally. Generated data is ignored by Git.

- 2026-05-29: 小改进
	- 将训练脚本 `src/swbaseline/train.py` 中零散的 `print` 替换为 `logging`，并在训练循环加入 `tqdm` 进度条以便监控训练进度。
	- 将数据审计脚本 `src/swbaseline/audit_data.py` 的输出替换为 `logging`。
	- 在 `requirements.txt` 中添加 `tqdm` 依赖。

