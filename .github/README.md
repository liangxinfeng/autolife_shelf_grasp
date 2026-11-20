# 367 抓取与任务流程

本分支对应 367 正式入口安装的 `grasp-2bd6726ec948ca6c`。从包根目录可找到 Python 实现、ROS 启动配置、机器人模型、测试、点选录入的垃圾清单、抓取标定、任务和固定动作库。已保存动作所用的 21 个视觉参考在 `data/fixed_actions/visual_references/`。大模型随仓库通过 Git LFS 分发。

## 取得并核对完整功能包

```bash
git lfs install
git clone --branch 367 --single-branch https://github.com/liangxinfeng/autolife_shelf_grasp.git
cd autolife_shelf_grasp
git lfs pull
python3 deployment/verify_367.py
```

`SOURCE_367.json` 提供每个文件的来源、权限、字节数和 SHA-256。标记 `installed` 的文件来自本次正式安装，`active-data` 是发布时的持久配置和已保存动作，`installed-model` 是同一安装使用的视觉与语音资源。`package-support` 包含源包附带的历史文档及辅助脚本。原始临时会话、上次动作计划及运行中的任务进度保存在原机。

## 运行环境

ROS 依赖见 `package.xml`。367 的启动入口 `start` 包含部署路径及控制中心的独占启动机制，配套环境脚本在 `deployment/367/`。迁移到其他设备时，应先配置 ROS 2 Jazzy、厂商 SDK、手眼标定、控制中心和导航等依赖，并核对底盘与关节接口。仓库中的程序包可在其源码目录运行，现场启动脚本以 367 的已安装路径为准。`rollback` 使用现场保留的上一版入口。

## 本版验证

已在隔离界面和运动模拟器运行桌面清理编排。纸杯在正立时录入一次，横放实录中沿用同一类别，另一个场景中无录入样本也能选出破损包装。整套回归收集 2303 项，2302 项通过，1 项因隔离环境缺少历史标定夹具而跳过。测试期间没有驱动真机。
