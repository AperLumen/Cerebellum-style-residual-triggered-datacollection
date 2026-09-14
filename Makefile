# CMCI 项目便捷入口。
#
# 真正的逻辑都在 scripts/check_all.py 里——这里只是薄封装，方便 `make check`
# 这类肌肉记忆。不做任何自己的判断，避免"两套验收标准"。

PY ?= .venv/bin/python

.PHONY: help check check-all fix test types pdf guide clean-guide \
        day-close drift-check drift-update hooks

help:
	@echo "make check      一键验收（快速失败）—— 提交前跑这个"
	@echo "make check-all  跑完全部步骤再汇总"
	@echo "make test       只跑测试"
	@echo "make types      只跑类型检查"
	@echo "make pdf        重建 GUIDE.pdf 并自检"
	@echo "make guide      只重建 GUIDE.pdf（不编译 LaTeX 的话加 --skip-pdf）"
	@echo "make clean-guide 删除 LaTeX 中间产物（guide.tex 等）"
	@echo ""
	@echo "用数据跑一次（引擎 CLI）："
	@echo "  .venv/bin/python scripts/cmci.py run <video> --imu <csv> --out ./out"
	@echo "  .venv/bin/python scripts/cmci.py run <video> --out ./out --dry-run"
	@echo ""
	@echo "协作闸门（防隐形改动 / 防幻觉，见 AGENTS.md）："
	@echo "  make day-close     每日收尾：make check 全绿 → 单提交 → 移基线 tag"
	@echo "  make drift-check   配置默认值漂移检查"
	@echo "  make drift-update  有意调参后更新基线（需先立 manifest 条目）"
	@echo "  make hooks         安装 pre-commit 快速闸门（漂移 + 溯源）"

check:
	@$(PY) scripts/check_all.py

check-all:
	@$(PY) scripts/check_all.py --all

test:
	@PYTHONPATH=. $(PY) -m pytest tests/ -q

types:
	@$(PY) -m pyright cerebellum_cancel/ tests/ scripts/

pdf:
	@$(PY) scripts/check_all.py --skip-pdf >/dev/null 2>&1 || true
	@$(PY) scripts/build_guide_pdf.py
	@$(PY) scripts/verify_guide_pdf.py

guide:
	@$(PY) scripts/build_guide_pdf.py

clean-guide:
	@rm -f guide.tex guide.aux guide.log guide.out guide.toc
	@echo "已清理 LaTeX 中间产物（GUIDE.pdf 保留）"

day-close:
	@$(PY) scripts/day_close.py $(ARGS)

drift-check:
	@$(PY) scripts/check_drift.py

drift-update:
	@$(PY) scripts/check_drift.py --update

hooks:
	@$(PY) scripts/install_hooks.py
