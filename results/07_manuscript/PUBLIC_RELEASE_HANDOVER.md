# GitHub / reproducibility 发布移交

日期：2026-09-05。本文件取代此前“空仓库、未公开”的状态描述。

## 已完成

- 公开推送至 https://github.com/cdhuangjin/lpcode-mstf-leakage-safe-detection 。首次代码/证据提交：`1b6a5b9f7f274b22a718b53219581d5f57a30792`。
- 全部研究模块和既有测试、Gate A–D配置/清单/逐运行记录、1800条消融、negative-pair、mechanism、图表源数据及绘图程序已包含；不是只上传README。
- 增加固定依赖、README、数据字典、逐Gate命令、上游获取与SHA-256、CITATION.cff和RIS引用导出。
- 新环境运行：423项研究测试通过，19条既有警告；4项发布检查通过；五图可重生成。未重跑正式实验。
- 匿名GitHub API读取README、验证脚本和冻结registry，逐文件字节比对通过；重新克隆后无训练审计通过。详情见 `../08_submission_audit/public_repository_verification.json`。
- 正式registry的20个文件和原始稿件快照均未改变。完整训练/特征/分区实现保持原样，发布适配仅涉及路径读取、包发现、pytest临时父目录与公开绘图检查/无损TIFF压缩。
- 当前主稿与补充中已删除 `TODO_PUBLIC_RELEASE`，加入真实commit固定链接；新增独立Code availability和第22条软件/数值证据引用。主PDF仍12页，补充2页，Word审阅版18页。

## 四个指定技能的处理

**nature-data**：区分第三方raw corpus、本研究run-level records和figure source data；提供获取路径、版本、变量单位、字典、校验和。没有虚构DOI或许可。

**nature-citation**：核对可用性声明到实际deposit/上游来源的对应，新增可检索的软件与数值证据引用，导出有完整有序作者的RIS。本轮不为了CNS标签添加与发布无关的文献。

**nature-statistics**：保持positive-class F1、跨单元等权平均、3个seed clusters、10000 bootstrap replicates、Gate D描述性和负消融；声明没有用发布测试冒充完整训练复现。

**nature-ref-verifier**：按技能要求将原21条分两组复核。未发现虚构引用或关键DOI错配；发现正式出版版本与预印本差异，均记录，未将单来源确认标成独立多来源一致。未擅自整体替换研究所依据的文献版本。

## 发布边界与仍需确认

**公开可访问已完成；无限制开源复用许可尚未确认。** 已向作者询问新增代码MIT、本人拥有权利的汇总数据CC BY 4.0方案，尚未收到明确许可选择，因此LICENSE暂保留权利，不能声称已授予MIT/CC BY。这不阻止按作者本次明确要求公开供审阅，但仍限制对“完全开放可复用”的声明。

上游LPcode未识别到LICENSE：原始数据和原始main.py从其官方固定commit获取，不整包重新发布；兼容实现继承关系已标明。第三方再许可不能由本项目的许可证代替。

未上传凭据、虚拟环境、重复总档案、作者私有提示词、第三方PDF/模板class、训练缓存或模型checkpoint。八个小型baseline pickle仅为作者已有数值指标输出，不是模型或raw snippets。无独立归档DOI；GitHub tag/commit不能冒充Zenodo永久保存。

## 后续操作

作者确认许可后，在新版本中应用许可并同步论文的许可说明；需要永久归档时再由有权账户实际登记DOI。不要为了消除剩余说明而伪造开放许可或归档记录。
