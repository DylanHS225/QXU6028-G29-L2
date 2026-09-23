# QXU6028 G29 L2 分析代码

这个公开仓库提供 L2 数值分析代码和运行说明。G29 的课程原始数据没有公开；请通过课程授权渠道获取同组的 `L2_data.csv`、`L2_meta.json` 和 `results_template.json`，放入同一目录。`manifest.json` 可一并放入。

```sh
python3 -m pip install -r requirements.txt
python3 reproduce.py --data /path/to/G29_dataset --output /path/to/new_empty_output
```

程序将从你提供的原始数据生成模型比较、能隙、费米能级轨迹、图表及诊断文件。公开仓库未包含原始数据，单独克隆它不能直接运行。`--checkpoint` 还要求原始数据在本地 Git 仓库中被追踪，此代码仓库本身不满足这一条件。

另行交付的 L2 压缩包保存了完整输入及输出。其 `results.json` 和 `commit_hash.txt` 会在核对线上源码提交后更新；Git 提交不能代替真实的小组贡献记录。
