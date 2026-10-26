# 一次启动，后台自动完成全部 ScanRefer 消融

## 你只需要执行一次

在已经配置好GLM、GPU环境和数据的服务器上，激活平时跑主实验的conda环境，然后解压本包。进入解压后的目录，执行：

```bash
python start_all_scanrefer.py --repo /root/SeqVLM-clean
```

把 `/root/SeqVLM-clean` 换成你实际的仓库路径。这个命令会自动读取该仓库 `config.yaml` 中的 `glm.model`，不需要手动填写模型占位符，不需要再追加执行/恢复参数，也没有API次数上限。请确保当前glm配置仍是你10月16日Full使用的实际型号。

包内已经包含你上传的Full参考日志，默认直接使用它，不需要重新移动日志。使用其他日志时可显式加 `--full-log /path/to/full.log`。

脚本会检查配置、备份安装新版runner并检查数据；已有不同版本runner会备份到这次输出目录。随后启动独立后台进程，输出PID及结果路径。正常关闭SSH窗口不会终止它；服务器关机、容器销毁等仍会终止进程。

这里只提供可在你服务器执行的启动包，没有连接或启动你的服务器。当前环境仅完成离线测试，真实推理必须在你有数据、模型权重和API配置的环境执行。

## 自动跑哪些组

默认按以下顺序连续运行，每组固定同一250条ScanRefer：

1. `anchor_target`：5张外观图 + anchor文本，不含关系图。
2. `local_full`：3张外观图 + 2张关系图，无最终全局验证。
3. `no_global_same_gate`：同样的局部证据，保留Full相同局部门控prompt，但不提供全局图。
4. `baseline`：固定canvas，无anchor-aware分支。
5. `anchor`：固定canvas + anchor文本。
6. `target`：5张外观图，无anchor-aware分支。

最终表格另外加入原Full日志这一行，**Full完全不重跑**。没有把依赖不成立、实际开关无效的组合硬凑成额外消融，也没有擅自加入NR3D付费实验。优先跑前三组可以先拿到论文的核心AR/FG对照；完整结束后才生成最终可用标记。

## 自动完成什么

- 冻结复用Full中的250份query解析，不重新付费解析。
- 保留现有点云、crop、metadata和旧实验产物。
- 根据有效配置和源文件指纹复用新canvas；不同消融输入不会误用旧canvas。
- 相同模型、相同提示、相同图像的API请求复用已成功响应。
- 每个case保存断点。遇到临时API传输错误且该case未得到预测时，不将网络失败写成错误预测；按30/60/120秒退避，最多自动恢复该组3次。
- 一组永久数据/代码错误不会阻止其他独立组继续运行；失败组不会被标记完成。401/403鉴权错误会停止后续组，避免反复使用无效凭据。
- 每完成一组更新 `summary.csv` 与 `summary.md`。
- 6组全部成功并各有250条时，自动生成 `ablation_table.tex` 和 `PAPER_READY.json`。

全部顺序执行，以避免同时生成同一缓存/重复发相同请求，也避免多个CLIP实例占满GPU。能否在你人工标注完之前跑完，取决于API速度、候选数和重试情况，不能保证固定结束时间。程序不会根据结果好坏自动挑最好的一次，也不会为了改善指标重跑已完成case。

## 结果在哪里

启动时会显示类似路径：

```text
/root/SeqVLM-clean/ablation_runs/all_scanrefer_YYYYMMDD_HHMMSS/
```

| 文件 | 用途 |
|---|---|
| `status.json` | 当前运行哪一组、是否全部完成或出现失败 |
| `background.log` | 全部组的启动、重试、结束状态 |
| `summary.csv` | 最终各组指标，可发回来更新论文 |
| `summary.md` | 便于直接阅读的总表 |
| `ablation_table.tex` | 全部完成后生成的LaTeX表格 |
| `PAPER_READY.json` | 存在且status=all_complete才代表6组完整结束 |
| `配置名/worker.log` | 每组完整运行进度，包括case完成情况 |
| `配置名/run/配置名/cases/*.json` | 每条预测框、完整精度IoU、缓存与API统计 |
| `配置名/run/manifest.json` | 该组代码、数据和配置指纹 |
| `配置名/run/api_usage.jsonl` | 该组实际新API传输尝试和tokens |
| `previous_scripts/` | 本次安装前的不同版本脚本备份 |

## 真正中断后怎么继续

从同一个解压目录运行，明确指定启动时显示的旧输出目录：

```bash
python start_all_scanrefer.py \
  --repo /root/SeqVLM-clean \
  --out /root/SeqVLM-clean/ablation_runs/all_scanrefer_YYYYMMDD_HHMMSS \
  --resume
```

不需要重审数据或重跑Full。已完成case与成功API响应会复用。若输入数据、源码或模型配置变化，脚本会拒绝伪装成同一实验恢复，请新开输出目录。不要在同一输出目录同时启动多个任务；后台进程有文件锁保护。

未填写 --out 的普通启动每次生成新目录，不会覆盖上次结果。因此不要为了查看进度反复运行启动命令，直接看status.json或日志即可。

## 使用结果前

确认 `PAPER_READY.json` 存在，并检查总表所有新跑行均为complete、n=250。若有失败，先看对应worker.log，再修复环境并恢复；不要把partial/failed的数字直接填进论文。

这些是一次固定运行的组件结果。Full来自历史日志；其准确GLM版本与历史prompt指纹无法从日志恢复，所以仍不能声称新旧运行完全消除了API随机性或远端模型版本变化。需要更细的组件解释时看附带的 `ADVANCED_README_zh.md`。
