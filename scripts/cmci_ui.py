#!/usr/bin/env python3
"""CMCI 本地前端 —— 浏览器界面，零依赖（只用 Python 标准库）。

    PYTHONPATH=. .venv/bin/python scripts/cmci_ui.py
    # 然后打开 http://127.0.0.1:8790

## 设计约束（不是为了省事，是为了不出错）

1. **前端自己不计算。** 它只调引擎的 `run()` 与 `replay()`。任何"顺手在这里
   算一下"都会分叉出第二个真值源——本项目已经在 public_edition 上踩过这个坑。
2. **零第三方依赖。** 给不写代码的人用，装不上就等于不存在。HTTP 用
   `http.server`，界面是纯 HTML/JS。
3. **数据不出本机。** 只监听 127.0.0.1，不引用任何外部资源（CDN 都不引），
   离线可用。
4. **调参走 replay，不重跑前端。** 实测重跑成本 79% 在前端前向，而 score
   与触发参数无关，所以调阈值时只重放触发层（实测 1907x 加速）。

## 为什么"调参"是安全的

`replay()` 只读引擎产出的无损 `score`（float64 决策标量），不碰 z 图，
也不重算任何前端结果。所以调参**不可能**产生一个与 CLI 不同的答案——
两条路径共用同一个引擎。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cerebellum_cancel.contracts import ContractError          # noqa: E402
from cerebellum_cancel.core import TriggerConfig               # noqa: E402
from cerebellum_cancel.engine import (EngineConfig, replay,     # noqa: E402
                                      run)

# 会话状态：**只缓存引擎的产物**，不缓存任何前端自己算的东西。
# 单用户本地工具，用全局 dict 足够（不引入会话管理这种复杂度）。
_SESSION: dict = {}


# ----------------------------------------------------------------- 业务动作

def _resolve(path: str) -> str:
    """把用户输入的路径规整成绝对路径，并给出可读的错误。"""
    p = os.path.expanduser(path.strip())
    if not os.path.isabs(p):
        p = os.path.join(ROOT, p)
    return os.path.normpath(p)


def act_run(payload: dict) -> dict:
    video = _resolve(payload.get("video", ""))
    imu = payload.get("imu") or None
    imu = _resolve(imu) if imu else None

    if not os.path.exists(video):
        return {"ok": False, "error": f"找不到视频文件：{video}"}
    if imu and not os.path.exists(imu):
        return {"ok": False, "error": f"找不到 IMU 文件：{imu}"}

    cfg = EngineConfig(
        fps=payload.get("fps") or None,
        imu_offset=float(payload.get("imu_offset") or 0.0),
        mirror=bool(payload.get("mirror")),
        max_frames=payload.get("max_frames") or None,
        keep_z_maps=True,
    )
    try:
        res = run(video, imu, cfg=cfg)
    except ContractError as exc:
        return {"ok": False, "error": f"输入契约不满足：{type(exc).__name__}: {exc}"}
    except OSError as exc:
        return {"ok": False, "error": f"读取失败：{exc}（路径是帧目录还是视频文件？）"}

    _SESSION["res"] = res
    _SESSION["video"] = video
    _SESSION["imu"] = imu
    return {"ok": True, **_describe(res)}


def act_replay(payload: dict) -> dict:
    """调参重跑：只重放触发层（不重跑前端）。"""
    res = _SESSION.get("res")
    if res is None:
        return {"ok": False, "error": "还没有跑过任何录制，请先点「开始过滤」。"}

    def num(key, default, lo, hi):
        try:
            v = float(payload.get(key, default))
        except (TypeError, ValueError):
            return default
        return min(max(v, lo), hi)

    cfg = TriggerConfig(
        threshold_k=num("threshold_k", 6.0, 1.0, 60.0),
        off_k=num("off_k", 2.5, 0.0, 59.0),
        pre_roll=num("pre_roll", 0.3, 0.0, 10.0),
        post_roll=num("post_roll", 0.8, 0.0, 10.0),
        min_clip_len=num("min_clip_len", 0.4, 0.0, 30.0),
        robust=bool(payload.get("robust")),
    )
    # off_k 必须 < threshold_k，否则回滞无意义。这里明确拒绝而不是悄悄纠正——
    # 悄悄纠正会让用户以为参数生效了。
    if cfg.off_k >= cfg.threshold_k:
        return {"ok": False,
                "error": f"关闭阈值 off_k ({cfg.off_k}) 必须小于开启阈值 "
                         f"threshold_k ({cfg.threshold_k})。"}

    r2 = replay(res, trigger_cfg=cfg)
    _SESSION["res"] = r2
    return {"ok": True, **_describe(r2)}


def act_export(payload: dict) -> dict:
    res = _SESSION.get("res")
    if res is None:
        return {"ok": False, "error": "还没有跑过任何录制。"}
    out = _resolve(payload.get("out") or os.path.join(ROOT, "cmci_out"))
    fmts = payload.get("formats") or ["dataset", "html"]
    # parquet 需要 pyarrow，缺了要说清楚而不是静默跳过
    if "parquet" in fmts:
        try:
            import pyarrow  # type: ignore  # noqa: F401
        except ImportError:
            return {"ok": False,
                    "error": "导出 parquet 需要 pyarrow（pip install pyarrow），"
                             "或者把 parquet 从格式里去掉。"}
    try:
        written = res.save(out, formats=tuple(fmts))
    except ContractError as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "written": {k: os.path.abspath(v)
                                    for k, v in written.items()}}


def _describe(res) -> dict:
    """把结果转成前端要的 JSON。**只读，不做任何加工计算。**"""
    s = res.summary()
    return {
        "summary": s,
        "segments": [seg.as_dict() for seg in res.segments],
        "curve": _downsample(res.surprise, 600),
        "keep_bands": _bands(res.keep, res.t),
        "meta": res.meta,
        "has_imu": bool(_SESSION.get("imu")),
    }


def _downsample(a, n: int) -> list:
    """降采样用于绘图（纯展示，不参与任何决策）。"""
    import numpy as np
    if len(a) <= n:
        return [float(x) for x in a]
    idx = np.linspace(0, len(a) - 1, n).astype(int)
    return [float(a[i]) for i in idx]


def _bands(keep, t) -> list:
    """keep=1 的连续区间，供前端画色带。"""
    import numpy as np
    out, i, n = [], 0, len(keep)
    while i < n:
        if keep[i]:
            j = i
            while j + 1 < n and keep[j + 1]:
                j += 1
            out.append([float(t[i]), float(t[j])])
            i = j + 1
        else:
            i += 1
    return out


# ------------------------------------------------------------------- HTTP

PAGE = r"""<!DOCTYPE html>
<html lang="zh"><meta charset="utf-8">
<title>CMCI 过滤工具</title>
<style>
 :root{--bd:#e3e3e3;--fg:#1c1c1c;--mut:#777;--acc:#0b5;--warn:#b80}
 *{box-sizing:border-box}
 body{font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
      margin:0;color:var(--fg)}
 header{padding:1rem 1.6rem;border-bottom:1px solid var(--bd);background:#fafafa}
 header h1{margin:0;font-size:1.1rem}
 header p{margin:.3rem 0 0;color:var(--mut);font-size:.9em}
 main{display:flex;gap:0;align-items:flex-start}
 #panel{width:340px;flex:0 0 340px;padding:1.2rem 1.6rem;
        border-right:1px solid var(--bd);min-height:70vh}
 #view{flex:1;padding:1.2rem 1.6rem;min-width:0}
 h2{font-size:.95rem;margin:1.2rem 0 .5rem;color:#333}
 label{display:block;margin:.7rem 0 .2rem;font-size:.86em;color:#444}
 input[type=text],input[type=number]{width:100%;padding:.42rem .5rem;
      border:1px solid var(--bd);border-radius:5px;font:inherit;font-size:.92em}
 input[type=range]{width:100%}
 .row{display:flex;gap:.5rem;align-items:center}
 button{padding:.5rem .9rem;border:1px solid var(--bd);border-radius:6px;
        background:#fff;font:inherit;cursor:pointer}
 button.primary{background:var(--acc);color:#fff;border-color:var(--acc);
        font-weight:600}
 button:disabled{opacity:.5;cursor:default}
 .kpi{display:flex;gap:.8rem;flex-wrap:wrap;margin:.6rem 0 1rem}
 .kpi div{background:#fafafa;border:1px solid var(--bd);border-radius:7px;
      padding:.5rem .8rem;min-width:6.4rem}
 .kpi b{display:block;font-size:1.25rem}
 .kpi span{color:var(--mut);font-size:.8em}
 table{border-collapse:collapse;width:100%;font-size:.88em}
 th,td{border-bottom:1px solid var(--bd);padding:.34rem .5rem;text-align:right}
 th:first-child,td:first-child{text-align:left}
 code{background:#f2f2f2;padding:.08rem .3rem;border-radius:3px;font-size:.88em}
 .note{background:#fffbe6;border-left:3px solid #f0c000;padding:.65rem .9rem;
      font-size:.88em;margin:.9rem 0;border-radius:0 5px 5px 0}
 .err{background:#fdecea;border-left:3px solid #d33;padding:.65rem .9rem;
      font-size:.9em;margin:.9rem 0;border-radius:0 5px 5px 0;white-space:pre-wrap}
 .mut{color:var(--mut);font-size:.86em}
 #chartWrap{position:relative;border:1px solid var(--bd);border-radius:7px;
      background:#fcfcfc;margin-top:.5rem}
 #tip{position:absolute;pointer-events:none;background:#222;color:#fff;
      padding:.2rem .45rem;border-radius:4px;font-size:.78em;display:none;
      white-space:nowrap;z-index:5}
 .slider{display:flex;align-items:center;gap:.5rem}
 .slider input{flex:1}
 .slider output{width:3.4rem;text-align:right;font-variant-numeric:tabular-nums}
 fieldset{border:1px solid var(--bd);border-radius:7px;padding:.6rem .8rem;
      margin:1rem 0}
 legend{font-size:.84em;color:#444;padding:0 .3rem}
</style>
<header>
 <h1>CMCI 过滤工具</h1>
 <p>把一段录制（视频 + IMU）过滤成数据集。全部在本机运行，不联网。</p>
</header>
<main>
 <div id="panel">
  <h2>1. 输入</h2>
  <label>视频文件路径</label>
  <input id="video" type="text" placeholder="data/webcam_rec/video.mp4">
  <label>IMU csv 路径（可留空）</label>
  <input id="imu" type="text" placeholder="imu.csv">
  <label>只处理前 N 帧（可留空 = 全部）</label>
  <input id="max_frames" type="number" placeholder="例如 900">
  <div class="row" style="margin-top:1rem">
   <button id="go" class="primary">开始过滤</button>
   <span id="status" class="mut"></span>
  </div>

  <fieldset id="tuneBox" disabled>
   <legend>2. 调参（改完即时重跑）</legend>
   <label>开启阈值 threshold_k</label>
   <div class="slider"><input id="threshold_k" type="range" min="2" max="40"
        step="0.5" value="6"><output id="threshold_k_o">6.0</output></div>
   <label>关闭阈值 off_k（须小于开启值）</label>
   <div class="slider"><input id="off_k" type="range" min="0" max="30"
        step="0.5" value="2.5"><output id="off_k_o">2.5</output></div>
   <label>片段前延 pre_roll（秒）</label>
   <div class="slider"><input id="pre_roll" type="range" min="0" max="3"
        step="0.1" value="0.3"><output id="pre_roll_o">0.3</output></div>
   <label>片段后延 post_roll（秒）</label>
   <div class="slider"><input id="post_roll" type="range" min="0" max="3"
        step="0.1" value="0.8"><output id="post_roll_o">0.8</output></div>
   <label><input id="robust" type="checkbox"> 稳健统计（密集 surprise 下不锁死）</label>
   <div class="mut" id="rt"></div>
  </fieldset>

  <fieldset id="outBox" disabled>
   <legend>3. 导出</legend>
   <label>输出目录</label>
   <input id="out" type="text" value="./cmci_out">
   <div class="row" style="flex-wrap:wrap;gap:.6rem;margin-top:.5rem">
    <label style="margin:0"><input type="checkbox" class="fmt" value="dataset" checked> dataset</label>
    <label style="margin:0"><input type="checkbox" class="fmt" value="html" checked> html</label>
    <label style="margin:0"><input type="checkbox" class="fmt" value="jsonl"> jsonl</label>
    <label style="margin:0"><input type="checkbox" class="fmt" value="parquet"> parquet</label>
   </div>
   <div class="row" style="margin-top:.7rem">
    <button id="exp">导出</button><span id="expStatus" class="mut"></span>
   </div>
  </fieldset>
 </div>

 <div id="view">
  <div id="msg" class="mut">在左边填好路径，点「开始过滤」。</div>
  <div id="result" style="display:none">
   <div class="kpi" id="kpi"></div>
   <div id="honest" class="note"></div>
   <h2>残差曲线与选中区间</h2>
   <div id="chartWrap"><div id="tip"></div>
     <svg id="chart" viewBox="0 0 1000 150" width="100%"
          preserveAspectRatio="none" style="height:150px;display:block"></svg>
   </div>
   <div class="mut" style="margin-top:.3rem">把鼠标移到曲线上可看该帧的残差强度。</div>
   <h2>片段</h2>
   <div id="segTable"></div>
  </div>
 </div>
</main>
<script>
const $ = id => document.getElementById(id);
let CUR = null, CURVE = null;

function post(url, body) {
  return fetch(url, {method:'POST', headers:{'Content-Type':'application/json'},
                     body: JSON.stringify(body)}).then(r => r.json());
}
function fmt(x, n=2){ return (x===null||x===undefined)?'—':Number(x).toFixed(n); }

// ---- 调参滑块：拖动即时重跑（后端只重放触发层，约 2 ms）
const SLIDERS = ['threshold_k','off_k','pre_roll','post_roll'];
SLIDERS.forEach(k => {
  $(k).addEventListener('input', () => { $(k+'_o').value = fmt($(k).value,1); });
});
let timer = null;
function scheduleReplay(){
  clearTimeout(timer);
  timer = setTimeout(doReplay, 60);   // 防抖，避免拖动时打满
}
SLIDERS.forEach(k => $(k).addEventListener('input', scheduleReplay));
$('robust').addEventListener('change', scheduleReplay);

function params(){
  const p = {};
  SLIDERS.forEach(k => p[k] = parseFloat($(k).value));
  p.robust = $('robust').checked;
  return p;
}

async function doReplay(){
  if(!CUR) return;
  const r = await post('/api/replay', params());
  if(!r.ok){ showErr(r.error); return; }
  clearErr(); render(r); $('rt').textContent = '已重跑（只重放触发层）';
}

async function doRun(){
  $('status').textContent = '跑着…';
  $('go').disabled = true;
  const body = {video:$('video').value, imu:$('imu').value,
                max_frames: parseInt($('max_frames').value)||null};
  if(!body.video){ $('status').textContent=''; $('go').disabled=false;
                   showErr('请先填视频路径。'); return; }
  const r = await post('/api/run', body);
  $('status').textContent = ''; $('go').disabled = false;
  if(!r.ok){ showErr(r.error); return; }
  clearErr(); CUR = r;
  $('tuneBox').disabled = false; $('outBox').disabled = false;
  render(r);
  if(!r.has_imu) $('rt').textContent = '本次没有 IMU（已退化，见下方说明）';
  else $('rt').textContent = '';
}

function render(r){
  CUR = r;
  const s = r.summary;
  $('msg').style.display='none'; $('result').style.display='block';
  $('kpi').innerHTML = [
    [s.n_frames,'帧'],[fmt(s.duration_s,1),'秒'],[s.n_segments,'片段'],
    [fmt(s.keep_fraction*100,1)+'%','帧级保留'],
    [fmt(s.stored_seconds,1)+'s','片段总时长']
  ].map(([v,k])=>`<div><b>${v}</b><span>${k}</span></div>`).join('');

  // 诚实说明：这一段必须始终可见，且随"有没有 IMU"变化
  const imuNote = r.has_imu
    ? '本次使用了 IMU：设备自身的旋转已被抵消。'
    : '<b>本次没有 IMU</b>，已退化为不补偿模式——任何晃动都会被当成"变化"，'
      + '筛选质量明显更低。';
  $('honest').innerHTML =
    `<b>选择依据。</b>本结果由<b>第 1 层「运动去重」</b>产生：选中"设备自身运动`
    + `预测不了"的帧。这<b>不等于</b>语义重要——后者在本项目的真实数据上已被否证。`
    + `<br>${imuNote}`
    + `<br>片段标记：${[...new Set(r.segments.map(x=>x.selection_reason))]
        .map(x=>`<code>${x}</code>`).join(' ') || '（无片段）'}`;

  drawChart(r); drawTable(r);
}

function drawChart(r){
  CURVE = r;
  const W=1000, H=150, c=r.curve, n=c.length;
  if(n<2){ $('chart').innerHTML=''; return; }
  const mn=Math.min(...c), mx=Math.max(...c), sp=(mx-mn)||1;
  const x = i => i/(n-1)*W, y = v => H - (v-mn)/sp*(H-14) - 7;
  const pts = c.map((v,i)=>`${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  // keep 色带：按时间比例映射到绘图宽度
  const t0=r.summary.duration_s? 0:0;
  const total = r.curve.length;
  const tmax = r.keep_bands.length? Math.max(...r.keep_bands.map(b=>b[1])) : 1;
  const bands = r.keep_bands.map(([a,b])=>{
    const xa = r.summary.duration_s ? (a/r.summary.duration_s)*W : 0;
    const xb = r.summary.duration_s ? (b/r.summary.duration_s)*W : 0;
    return `<rect x="${xa.toFixed(1)}" y="${H-9}" width="${Math.max(xb-xa,1).toFixed(1)}"
             height="9" fill="#0b5" opacity="0.5"/>`;
  }).join('');
  $('chart').innerHTML =
    bands + `<polyline points="${pts}" fill="none" stroke="#c33" stroke-width="1.3"/>`
    + `<text x="6" y="14" font-size="12" fill="#888">残差强度</text>`
    + `<text x="${W-96}" y="${H-2}" font-size="12" fill="#0b5">选中区间</text>`;
  // 悬停提示
  const wrap=$('chartWrap'), tip=$('tip');
  $('chart').onmousemove = ev=>{
    const rc=wrap.getBoundingClientRect();
    const rel=(ev.clientX-rc.left)/rc.width;
    const i=Math.max(0,Math.min(n-1,Math.round(rel*(n-1))));
    const t= r.summary.duration_s ? rel*r.summary.duration_s : 0;
    tip.style.display='block';
    tip.style.left=Math.min(ev.clientX-rc.left+10, rc.width-120)+'px';
    tip.style.top=(ev.clientY-rc.top-26)+'px';
    tip.textContent=`t≈${t.toFixed(2)}s  残差=${c[i].toFixed(3)}`;
  };
  $('chart').onmouseleave = ()=> tip.style.display='none';
}

function drawTable(r){
  if(!r.segments.length){
    $('segTable').innerHTML =
      '<p class="mut">本次没有选出任何片段。这通常意味着整段录制里没有"设备自身'
      + '运动解释不了"的变化（例如全程静止）。它<b>不</b>表示数据有问题。</p>';
    return;
  }
  const rows = r.segments.map((s,i)=>`<tr>
    <td>${i}</td><td>${fmt(s.start)}</td><td>${fmt(s.end)}</td>
    <td>${fmt(s.duration)}</td><td>${s.n_frames}</td>
    <td>${fmt(s.mean_score,1)}</td><td>${fmt(s.peak_score,1)}</td>
    <td><code>${s.selection_reason}</code></td></tr>`).join('');
  $('segTable').innerHTML = `<table><tr><th>#</th><th>起点(s)</th><th>终点(s)</th>
    <th>时长(s)</th><th>帧数</th><th>均值</th><th>峰值</th><th>选择依据</th></tr>
    ${rows}</table>`;
}

function showErr(m){
  clearErr();
  const d=document.createElement('div'); d.className='err'; d.textContent=m;
  $('view').prepend(d);
}
function clearErr(){ document.querySelectorAll('.err').forEach(e=>e.remove()); }

$('go').onclick = doRun;
$('exp').onclick = async ()=>{
  const formats=[...document.querySelectorAll('.fmt:checked')].map(x=>x.value);
  if(!formats.length){ $('expStatus').textContent='请至少选一种格式'; return; }
  $('expStatus').textContent='导出中…';
  const r = await post('/api/export', {out:$('out').value, formats});
  $('expStatus').textContent = r.ok ? '已导出' : '失败';
  if(!r.ok){ showErr(r.error); return; }
  clearErr();
  const lines = Object.entries(r.written)
    .map(([k,v])=>`<div>${k}: <code>${v}</code></div>`).join('');
  const d=document.createElement('div'); d.className='note';
  d.innerHTML = `<b>导出完成</b>${lines}`;
  $('view').prepend(d);
};
</script>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """静音访问日志（本地工具，不需要）。

        参数名必须是 `format`（与基类一致）——基类这里刻意遮蔽了内置
        `format`，改名字会让 pyright 报 override 不兼容。
        """
        return None

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # 本地工具：不缓存，避免改代码后看到旧界面
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):                                  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self):                                 # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            self._send(400, b'{"ok":false,"error":"bad json"}',
                       "application/json")
            return

        actions = {"/api/run": act_run, "/api/replay": act_replay,
                   "/api/export": act_export}
        fn = actions.get(self.path)
        if fn is None:
            self._send(404, b'{"ok":false,"error":"unknown endpoint"}',
                       "application/json")
            return
        try:
            out = fn(payload)
        except Exception as exc:                       # 兜底：不要让界面白屏
            traceback.print_exc()
            out = {"ok": False,
                   "error": f"内部错误：{type(exc).__name__}: {exc}"}
        self._send(200, json.dumps(out, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="CMCI 本地前端")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--host", default="127.0.0.1",
                    help="默认只监听本机（数据不出本机）")
    args = ap.parse_args()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"CMCI 前端已启动： http://{args.host}:{args.port}")
    print("  · 只监听本机，数据不出网")
    print("  · 调参走 replay（只重放触发层），不重跑前端")
    print("  · Ctrl-C 退出")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
