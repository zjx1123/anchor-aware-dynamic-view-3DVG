> 本文件为高级手动用法和代码审计说明。一键自动跑全部消融请优先阅读 README_zh.md，并运行 start_all_scanrefer.py。

# AnchorVRA：ScanRefer 消融重跑方案

核对版本：仓库提交 `fa5d69b51bef8ef8fb3e061dfd13606c626ab07c`（2026-10-23）。
Full 使用用户上传的 `scanrefer_full_20261016_220015(2).log`；用户已确认它属于 GLM 系列，主实验使用 3 appearance + 2 relation。

## 结论

旧消融的六行设计可沿用，但目前的批量脚本不能原样运行。新版独立调用当前 `AdpativePredictor`，不覆盖主流程和旧消融文件。Full 只导入日志，程序没有执行 Full 的选项。

已离线核对250条日志与 `data/scanrefer_250.json` 的场景、目标ID、目标类别、caption及顺序，全部一致；从 Full 日志恢复了250份实际使用的 query parse。objprint 对含撇号的 raw_query 打印不符合 Python 字面量格式，因此恢复时只用已匹配的数据集 caption 替换这个冗余字段，再用 `ast.literal_eval` 读取其余语义字段，不重新调用模型解析。

Full 行：Overall@25=58.4、Overall@50=50.8、Unique@25=81.8、Unique@50=74.2、Multiple@25=50.0、Multiple@50=42.4（百分比，保持原日志输出精度）。250条；异常0；210条有最终局部门控日志，其中15条实际构造最终全局图。860段 VLM 输出是日志可见数量，不等同于包含失败请求的完整计费账单。

## 旧代码的具体问题

| 文件/位置（按函数或关键字定位） | 发现 | 新版处理 |
|---|---|---|
| `seqvlm/run_ablation.py` → `DATASET_ARGS['scanrefer']['entry']` | 指向 `evaluate.py` | 新版直接实例化当前 predictor，不依赖错误入口 |
| `run_ablation.py` → `build_command()` | 向 `evaluate.py` 传 `--seg_conf_score` 和 `--use_anchor_aware`；该入口没有这些参数 | 新版显式传入 predictor 的 AQ、阈值和各视图参数 |
| `seqvlm/evaluate.py` → `vlm_configs['use_anchor_aware'] = True` | Anchor 被写死开启，不能按它实现关闭 AQ 的 Baseline/Target 行 | AQ关时不执行 query/anchor 分支 |
| `seqvlm/evaluate_ablation.py` | 已有可控参数，但旧 runner 没有调用它 | 最小修复是改入口；更完整方案使用本包新 runner |
| `run_ablation.py` → `parse_args()` 与 `run_script.sh` | 旧 runner 默认 qwen，当前主脚本 glm | 新版默认 glm，并要求付费执行时用 `--expected-model` 核对 config.yaml 中具体模型名称 |
| `dynamic_view_selector.py` → `_query_hash()` / `build_query_specific_canvas()` | 缓存只含 query+target，不含K、appearance/relation数量、anchor输入、源图/metadata版本 | 新缓存包含配置、实际上下游依赖及源图内容指纹，禁止拿AQ/AR变更前的canvas冒充新结果 |
| `dynamic_view_selector.py` → `use_global_context` | 只保存变量，后续未使用 | 不将它列作独立组件；最终全局模块仍由 `use_final_global_view` 控制 |
| `query_parser.py` → `_cache_key()` | 旧缓存仅以文本MD5命名，不含解析模型和prompt版本 | 本次冻结Full日志中的解析结果；不扫描/覆盖旧缓存，也不重新解析 |
| `adaptive_predictor.py` → `predict()` 最后一轮 | local_full用normal prompt；Full用final_local_gate prompt，必要时再global decision | 对比两行表示最终验证模块整体差异；若要更严格研究全局图，额外跑 `no_global_same_gate` |
| `run_ablation.py` → `mkdir(...exist_ok=True)` 与 `run_and_tee(...'w')` | 同output_root可能覆盖日志，无逐case断点 | 新版禁止无说明复用目录；`--resume` 校验代码/参数/输入指纹后恢复 |

当前实现还需要与论文表述对齐：appearance 是固定质量分数排序，不是额外调用VLM逐帧打分；relation选择器按共可见性与质量为最高置信度anchor选帧。Anchor属性/关系被保留在文本提示里，`build_anchor_candidates` 没有在几何阶段严格求解“最左/最近”等约束。TAR可以作为这些关联记录的形式化定义，但不宜宣称实现了尚未编码的属性筛选或完备空间约束验证。

## 配置与推荐顺序

AQ代表目标—参照物解析、实例关联和anchor-aware文本推理整体，TC代表目标外观证据构建（质量排序、红框和canvas），AR代表目标—参照物同屏关系图，FG代表最终全局验证模块。

| 名称 | AQ | TC | AR | FG | 实际图像 |
|---|---|---|---|---|---|
| baseline | 关 | 关 | 关 | 关 | 原固定canvas |
| anchor | 开 | 关 | 关 | 关 | 原固定canvas + anchor文本 |
| target | 关 | 开 | 关 | 关 | 最多5张appearance |
| anchor_target | 开 | 开 | 关 | 关 | 最多5张appearance + anchor文本 |
| local_full | 开 | 开 | 开 | 关 | 默认3 appearance + 2 relation，缺关系图时外观补位 |
| Full | 开 | 开 | 开 | 开 | **不重跑，直接导入10月16日日志** |
| no_global_same_gate（可选） | 开 | 开 | 开 | 无全局证据 | 保留Full相同局部门控prompt，但强制采用局部选择，不构造/调用全局分支 |

省钱可先跑 `anchor_target,local_full`，检查最核心的关系图组件，再决定是否补齐前面三行。完整重做原消融表则运行默认五组。若主文重点要分离“全局图”作用，再追加 `no_global_same_gate`；它可以共享 local_full 的canvas以及相同的中间API请求缓存。

不同AQ开关会改变提示词结构，因此AQ是组合组件消融。不同TC开关还改变视觉标记与canvas，不能把差值仅解释为某一条排序公式的作用。Full是独立历史运行，不能声称排除了远端模型采样、版本更新或历史prompt差异。

## 如何安装

1. 解压本包，把 `seqvlm/reablate_scanrefer.py` 复制到你服务器仓库的 `seqvlm/` 目录。旧 `evaluate.py`、`evaluate_ablation.py`、`run_ablation.py` 都不用覆盖。
2. 把包里的 `reference/full_scanrefer_20261016.log` 放在仓库 `logs/`，或使用自己已有日志的绝对路径。
3. 保留你当前 `config.yaml` 中 GLM 配置。只确认 `glm.model` 的具体值与10月16日Full实际模型相同；不要把密钥发出来。本包不含密钥。
4. 所有命令从仓库根目录执行。GPU/点云/CLIP/metadata/RGB仍使用原服务器已有数据；这里没有替你启动任何付费运行。

本仓库若只是从GitHub重新clone，许多 `data/*` 是指向 `/root/autodl-tmp/...` 的软链接。路径在另一台机器上可能失效，应在真正有数据的工作目录运行；脚本会先检查115个相关场景的主要文件。

## 第一步：离线检查（零API）

替换下面模型名为 `config.yaml` 的 `glm.model` 真实值。它不是 `glm` 这个别名。当前日志不记录具体模型版本，不能仅凭“GLM系列”自动推断。

```bash
python seqvlm/reablate_scanrefer.py \
  --full-log logs/full_scanrefer_20261016.log \
  --expected-model "YOUR_EXACT_GLM_MODEL_NAME" \
  --variants anchor_target,local_full \
  --out ablation_runs/scanrefer_ar_check
```

默认仅规划，生成 `manifest.json`、`preflight.json`、`query_parse_frozen.json`、Full行和待跑行的 `summary.csv/summary.md`。不导入推理模型，不发API。

检查 `preflight.json` 的 errors。数据缺失、模型名称不一致、未填具体模型等会阻止真正执行。若参数/模型/输入文件变了，请选新输出目录重新规划；不要给原目录硬加 `--resume`。

## 第二步：执行两组核心消融

与第一步完全相同的参数，追加执行和断点恢复；默认不限制API调用次数：

```bash
python seqvlm/reablate_scanrefer.py \
  --full-log logs/full_scanrefer_20261016.log \
  --expected-model "YOUR_EXACT_GLM_MODEL_NAME" \
  --variants anchor_target,local_full \
  --out ablation_runs/scanrefer_ar_check \
  --execute --resume
```

按用户要求，默认没有API次数上限，不会因为累计调用达到1000次而停止。保留逐case断点和成功API响应缓存，意外中断后重复相同命令可以继续。

要一次生成完整旧表的五组，将两步命令中的 variants 改为：

```text
baseline,anchor,target,anchor_target,local_full
```

并使用新的 `--out ablation_runs/scanrefer_all_components`。缓存根目录相同，已完成核心两组对应的相同图像和API请求仍可复用；新目录会重建自己的case记录和总结，不改动前一个实验目录。

需要同门控提示的无全局对照时，在新输出目录选择：

```text
--variants no_global_same_gate
```

主实验三个appearance、两个relation已作为默认。关闭AR时用5张appearance，使画布上限仍为5。若有效视图不足，实际张数可能小于5；这是原selector行为。新脚本在动态canvas构建完全失败时直接停止，防止悄悄回退固定canvas却仍声称动态组件有效。修好缺图/metadata后，用新输出目录重启（成功API缓存仍可复用）。

## 可选冒烟测试

在独立输出目录加 `--max-samples 2`。即使程序退出0，表中也标为 partial，不把2条结果当作250条论文结果。数据准备正常且采用同参数时，可以从这个目录 `--resume` 并去掉max-samples继续全量。

本次已执行零API的8项测试：Full逐例对齐与解析恢复、错query拒绝、IoU精度边界、API缓存和可选预算断点、默认无API次数上限、不同模型缓存隔离、组件依赖与Full不可执行、canvas依赖变化/关闭AR后的可共享缓存。

未在当前环境执行真实推理：这里缺少服务器上的CLIP权重、Mask3D/GT文件与RGB软链接目标，也没有你的config.yaml。因此测试通过不等于服务器端250条已跑通；新脚本会在服务器先做预检。

## 缓存、恢复和文件保护

- Full query parse：来自Full日志，读入独立快照；AQ开时复用，AQ关时不使用。不存在的query直接报错，不偷偷付费解析。
- 点云、固定canvas、crop、metadata、RGB：读取原目录，不改写。
- 新动态canvas：放在 `data/cache/reablation_v1/canvases/`。配置、元数据、proposal、anchor/parse（AR开启时）、图像源内容变化都会改变缓存路径。AR关闭时，不因只改变anchor文本而重复渲染相同appearance canvas。
- 老canvas：因缺少足够配置溯源信息，不盲目导入其“query+target”旧缓存。首次为新namespace本地生成，之后同逻辑复用；渲染不调用VLM。
- API：以真实模型标识、endpoint指纹、API源码和完整请求（含图片实际base64内容）生成键，只有完全相同请求才复用。请求摘要与返回结果分开保存，不保存密钥。
- case：每完成一条原子写入 `配置名/cases/0000.json`。中断只重做未完成case；case中途成功的API响应也有缓存。
- 结果目录：默认禁止复用；`--resume` 必须代码、数据、设置与文件清单指纹一致。标注和proposal metadata用内容hash；大型源数据/权重额外用路径、大小、mtime清单防止普通文件变更后误用case断点。不要人为保留旧mtime来替换权重/GT内容；换数据应新开实验目录。
- 源码快照：只保存相关Python/prompt源码，不保存含密钥的config.yaml。
- 计费记录：`api_usage.jsonl` 只记录新网络尝试，缓存命中不再计费。case中的 elapsed_seconds 包括本次缓存影响，不能拿缓存运行时长与原Full日志直接比较推理效率。
- 随机性：设置Python/NumPy/Torch seed和进程启动前PYTHONHASHSEED；不擅自向当前API增加temperature/seed参数。当前api.py使用服务端采样默认值，不能保证重复请求本来就确定性。共享响应缓存是一种固定请求实现的复用，不是独立重复试验。

## 输出与论文

- `summary.csv/summary.md`：五组或选中组 + imported Full，百分比显示；只有complete和reused_full_log可整理入论文。
- `*/effective_config.json`：AQ/TC/AR、实际appearance/relation参数和模型别名。
- `*/cases/*.json`：完整精度IoU、预测框、unique/multiple、API实际新调用和命中数。
- `*/console.log`：该组独立控制台日志，恢复时追加不覆盖。
- `full_case_index.csv`：250条Full场景/目标/描述清单。
- `query_parse_frozen.json`：本次实际复用的解析结果。

Full日志不包含精确GLM模型版本、当时所有prompt源码hash、每张已缓存canvas生成配置等，因此这里只能确认样本、日志行为与用户已确认的3+2、GLM系列。不能证明今天远端GLM与历史Full完全同一版本。若服务器源码/prompt在10月16日之后有变化，最好恢复10月16日那份ScanRefer源码和prompt再运行；新脚本会记录你实际运行版本。不要为了让差值好看而试多次挑最优行，保留失败与恢复记录。

完成后，将摘要、manifest和各组metrics发回来再更新论文表；不要在消融尚未完成时继续填原旧指标，也不要将组件差值写成未经控制的单因素因果结论。
