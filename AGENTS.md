# 协作规则（人与 AI 共同遵守）

> 改代码之前先读这里。违反任何一条，`make check` 或 `make day-close`
> 会在机制层面拦下，不靠自觉。

## 一、防幻觉、防伪需求（最高优先级）

1. **没有需求条目的 .py 改动不允许发生。** 每改一个 .py 文件，先在
   `requirement_manifest.json` 立条目：`id` / `date` / `source`（用户原话，
   逐字引用）/ `summary` / `files`。`make check` 的"需求溯源检查"会拿
   变更清单逐一对账（基线：上次 day-close 之后的一切改动）。
2. **"测试全过"不构成改动理由。** 它只证明没改坏，不证明该改。改动是否
   成立，唯一判据是条目里的用户原话。AI 特别警惕：不要"顺手优化""顺手
   抽象"——那是伪需求的标准长相。拿不准就先问。
3. **配置默认值是测量基础设施，不是可调旋钮。** `config_baseline.json`
   锁死了 `cerebellum_cancel/` 全部 dataclass 默认值；`make check` 逐字节
   比对。确需调参 = 改代码 + 立 manifest 条目 + `make drift-update`，
   三者缺一不可。

## 二、可追溯：每日收尾

4. 每个工作日结束跑 `make day-close`：强制 `make check` 全绿 → 生成
   `daily/YYYY-MM-DD.md` → 当天全部改动单提交 → 移动基线 tag。
   当天的所有改动必须在同一个提交里；提交信息禁模糊词（"优化""重构"），
   必须写清改了什么、对应哪条台账。
5. 想每次提交都被快速闸门检查，跑 `make hooks` 装 pre-commit 钩子
   （配置漂移 + 需求溯源，<1s）。卸载：`rm .git/hooks/pre-commit`。

## 三、执行口径

6. Python 一律 `PYTHONPATH=. .venv/bin/python ...`；不用 `X | Y` 运行时
   类型注解（解释器 3.9）。
7. 基线：pytest 537+ passed（ERROR 6 = dino 合成帧缺失，台账基线非失败）；
   `make check` 13 步全绿。
8. 验收只有一套：`make check`。Makefile 只是薄封装，不许另立标准。
