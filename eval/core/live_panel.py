"""LivePanel：后台 daemon 线程终端实时面板。

核心特性：
    - 后台 daemon 线程每 2 秒持锁浅拷贝 metrics → 格式化面板 → 清屏重绘（ANSI）或追加（plain）
    - ANSI/plain 双模式，通过 .env 中 LIVE_PANEL_MODE 切换，兼容 PyCharm embedded terminal
    - 面板显示：进度条 + Layer 1/Layer 2 指标 vs 上次 delta + 阶段延迟 P50/P75/P95 + 成本/缓存命中率 + 异常事件滚动区
    - 事件区仅显示异常（429 / parse_error / judge_error），不含 pass——成功的 query 不产生事件
    - render_final() 跑完后打印完整最终报告，替代老旧散落 print
    - stop() 幂等，finally 块安全调用
    - 模块级单例 get_panel() / set_panel()，深层 judge 代码直接 push_alert() 避免参数层层透传
    - push_alert() 无锁（deque.appendleft 在 CPython GIL 下原子）

用法示例::

    from eval.core.live_panel import LivePanel, set_panel
    panel = LivePanel(metrics, previous_per_query)
    set_panel(panel)
    panel.set_total(len(items))
    panel.set_meta("bench.json", generator_model="deepseek-v4-flash", judge_model="deepseek-v4-pro")
    panel.start()
    # ... as_completed loop ...
    panel.stop()
    panel.render_final()

公共接口：
    - LivePanel: 终端面板类（start / stop / query_done / push_alert / render_final）
    - get_panel: 模块级单例获取
    - set_panel: 模块级单例设置
"""

import sys
import threading
import time
from collections import deque
from datetime import datetime

DEFAULT_REFRESH_SEC = 2
ALERT_MAXLEN = 20
ALERT_DISPLAY_N = 5
PROGRESS_BAR_WIDTH = 30

# ANSI escape codes
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_RED = "\033[31m"
C_CYAN = "\033[36m"
CLEAR_SCREEN = "\033[2J"
CURSOR_HOME = "\033[H"
CURSOR_HIDE = "\033[?25l"
CURSOR_SHOW = "\033[?25h"


class LivePanel:
    """后台 daemon 线程终端面板。持有 MonitorMetrics 引用 + 上次运行基线。"""

    def __init__(self, metrics: "MonitorMetrics", previous_per_query: dict | None = None):
        self.metrics = metrics
        self.previous_per_query = previous_per_query  # query_id -> per_query dict

        self.query_count: int = 0
        self.total_queries: int = 0
        self.start_time: float = 0.0

        self._running: bool = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._alerts: deque[str] = deque(maxlen=ALERT_MAXLEN)
        self._crashed: bool = False

        self._benchmark_name: str = ""
        self._generator_model: str = ""
        self._judge_model: str = ""
        self._eval_mode: str = "full"

        from config import LIVE_PANEL_MODE
        self._mode: str = LIVE_PANEL_MODE

        self._redis_available: bool = self._check_redis()
        self._first_render: bool = True

    @staticmethod
    def _check_redis() -> bool:
        try:
            from infra.cache import get_cache
            from infra.cache.noop_backend import NoopBackend
            return not isinstance(get_cache(), NoopBackend)
        except Exception:
            return False

    def set_total(self, total: int):
        self.total_queries = total

    def set_meta(self, benchmark_name: str, generator_model: str = "", judge_model: str = "",
                 eval_mode: str = "full"):
        self._benchmark_name = benchmark_name
        self._generator_model = generator_model
        self._judge_model = judge_model
        self._eval_mode = eval_mode

    def start(self):
        if self._running:
            return
        self._running = True
        self.start_time = time.time()
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """幂等停止。第二次调用直接返回。"""
        if not self._running:
            return
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        self._terminal_reset()

    def query_done(self):
        with self._lock:
            self.query_count += 1

    def push_alert(self, query_id: str, message: str):
        """入队一条警告事件。无锁——deque.appendleft 在 CPython GIL 下原子。"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._alerts.appendleft(f"{timestamp}  {query_id}  {message}")

    # ============================================================
    # Internal: render loop
    # ============================================================

    def _render_loop(self):
        try:
            while self._running:
                with self._lock:
                    l1 = self.metrics.layer1_means()
                    l2 = self.metrics.layer2_means()
                    stages = self.metrics.stage_percentiles()
                    verdicts = self.metrics.verdict_distribution()
                    deltas = self._compute_deltas(self.metrics)
                    qc = self.query_count
                    alerts = list(self._alerts)[:ALERT_DISPLAY_N]
                    alert_total = len(self._alerts)
                    overflow = max(0, alert_total - ALERT_MAXLEN)
                self._render(l1, l2, stages, verdicts, deltas, qc, alerts, overflow)
                time.sleep(DEFAULT_REFRESH_SEC)
        except Exception:
            self._crashed = True
            self._running = False
            import traceback
            traceback.print_exc()

    def _render(self, l1, l2, stages, verdicts, deltas, qc, alerts, overflow):
        if self._mode == "plain":
            self._render_plain(l1, l2, stages, verdicts, deltas, qc, alerts, overflow)
        else:
            self._render_ansi(l1, l2, stages, verdicts, deltas, qc, alerts, overflow)

    # ============================================================
    # ANSI mode
    # ============================================================

    def _render_ansi(self, l1, l2, stages, verdicts, deltas, qc, alerts, overflow):
        if self._first_render:
            sys.stdout.write(CURSOR_HIDE)
            self._first_render = False

        lines = [CLEAR_SCREEN + CURSOR_HOME]
        lines.extend(self._build_panel_lines(l1, l2, stages, verdicts, deltas, qc, alerts, overflow))
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()

    # ============================================================
    # Plain mode
    # ============================================================

    def _render_plain(self, l1, l2, stages, verdicts, deltas, qc, alerts, overflow):
        lines = ["\n" + "=" * 76]
        lines.extend(self._build_panel_lines(l1, l2, stages, verdicts, deltas, qc, alerts, overflow))
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()

    # ============================================================
    # Panel content builder
    # ============================================================

    def _build_panel_lines(self, l1, l2, stages, verdicts, deltas, qc, alerts, overflow):
        lines = []

        # Header
        redis_status = "connected" if self._redis_available else f"{C_RED}unavailable (no cache){C_RESET}"
        lines.append(f"{C_BOLD}{'═' * 76}{C_RESET}")
        lines.append(f"  slight_rag eval · {self._eval_mode} mode · {self._benchmark_name} · {self.total_queries} queries")
        lines.append(f"  GeneratorModel: {self._generator_model}  |  JudgeModel: {self._judge_model}")
        lines.append(f"  Redis: {redis_status}")
        lines.append(f"{C_BOLD}{'═' * 76}{C_RESET}")
        lines.append("")

        # Progress
        elapsed = time.time() - self.start_time
        done = min(qc, self.total_queries)
        ratio = done / self.total_queries if self.total_queries > 0 else 0
        filled = int(ratio * PROGRESS_BAR_WIDTH)
        bar = "█" * filled + "░" * (PROGRESS_BAR_WIDTH - filled)
        lines.append(f"  进度  {bar}  {done}/{self.total_queries} ({ratio:.0%})  [{self._format_time(elapsed)}]")
        lines.append("")

        # Layer 1 + Layer 2 side by side
        valid_l2_count = len([r for r in list(self.metrics.layer2_results) if r.faithfulness is not None])
        delta_label = ""
        if deltas and "_matching" in deltas:
            delta_label = f"Δ (N={deltas['_matching']} matching)"

        lines.append("  Layer 1                                 Layer 2"
                     + (f" ({valid_l2_count} valid)" if valid_l2_count else ""))
        lines.append("  ────────────                            ────────────")
        lines.append("           本次      vs 上次                        本次      vs 上次")

        l1_names = ["Recall@5", "Prec@5", "Hit@5", "MRR", "MAP@5", "NDCG@5"]
        l1_keys = ["recall_at_k", "precision_at_k", "hit_at_k", "mrr", "map_at_k", "ndcg_at_k"]
        l2_names = ["Faithfulness", "Answer Relev", "Context Prec", "Context Rec", "Answer Corr"]
        l2_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]

        for i in range(6):
            l1_val = self._fmt_val(l1.get(l1_keys[i]))
            l1_delta = self._fmt_delta(deltas.get(l1_keys[i]))
            left = f"  {l1_names[i]:<10s}  {l1_val:>8s}  {l1_delta:>8s}"

            right = ""
            if i < 5 and l2:
                l2_val = self._fmt_val(l2.get(l2_keys[i]))
                l2_delta = self._fmt_delta(deltas.get(l2_keys[i]))
                right = f"  {l2_names[i]:<14s}  {l2_val:>8s}  {l2_delta:>8s}"
            elif i == 5 and l2:
                # Verdict line
                v = verdicts
                right = (f"  {C_GREEN}pass:{v.get('pass', 0)}{C_RESET}"
                         f"  {C_YELLOW}partial:{v.get('partial', 0)}{C_RESET}"
                         f"  {C_RED}fail:{v.get('fail', 0)}{C_RESET}"
                         f"  error:{v.get('error', 0)}")

            if right:
                lines.append(f"{left}    {right}")
            else:
                lines.append(left)

        if delta_label:
            lines.append(f"  {' ' * 62}{delta_label}")
        lines.append("")

        # Stage latency + Cost side by side
        lines.append("  阶段延迟 (P50 / P75 / P95)                   成本")
        lines.append("  ────────────────────────────               ──────────────────")

        stage_labels = ["Retrieve", "Generate", "Judge Faithfulness", "Judge Quality"]
        stage_keys = ["retrieve", "generate", "judge_faithfulness", "judge_quality"]
        cost_labels = ["Token", "Cost", "Generator Cache", "Judge Cache"]
        cost_values = [
            f"{self.metrics.total_input_tokens:,} in / {self.metrics.total_output_tokens:,} out",
            f"¥{self.metrics.estimated_cost_usd:.2f}",
            (f"{self.metrics.generator_cache_hits} hit "
             f"({self.metrics.generator_cache_hit_rate:.1%})"),
            (f"{self.metrics.judge_cache_hits} hit "
             f"({self.metrics.judge_cache_hit_rate:.1%})"),
        ]

        for i in range(4):
            s = stages.get(stage_keys[i], {})
            stage_data = f"{int(s.get('p50', 0))} / {int(s.get('p75', 0))} / {int(s.get('p95', 0))} ms"
            left = f"  {stage_labels[i]:<20s}  {stage_data:<24s}"

            # Cost: only show meaningful values if Redis unavailable
            if not self._redis_available and i in (2, 3):
                cost_str = "—"
            else:
                cost_str = cost_values[i]
            right = f"  {cost_labels[i]:<16s}  {cost_str}"

            lines.append(f"{left} {right}")
        lines.append("")

        # Alerts
        alert_total = len(self._alerts)
        if overflow > 0:
            lines.append(f"  {C_RED}⚠ 事件 ({alert_total}/{ALERT_MAXLEN})  {overflow} 条未显示{C_RESET}")
        elif alert_total > 0:
            lines.append(f"  {C_YELLOW}⚠ 事件 ({alert_total}/{ALERT_MAXLEN}){C_RESET}")
        else:
            lines.append("  ⚠ 事件 (0)")
        for a in alerts:
            lines.append(f"  {a}")

        lines.append("")
        lines.append(f"{C_BOLD}{'═' * 76}{C_RESET}")
        return lines

    # ============================================================
    # Final report
    # ============================================================

    def render_final(self):
        """跑完后打印最终报告。从 metrics 读取全部数据。"""
        try:
            m = self.metrics
            l1 = m.layer1_means()
            l2 = m.layer2_means()
            stages = m.stage_percentiles()
            verdicts = m.verdict_distribution()
            deltas = self._compute_deltas(m)
            elapsed = time.time() - self.start_time if self.start_time > 0 else 0
            run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")

            valid = [r for r in list(m.layer2_results) if r.faithfulness is not None]
            valid_n = len(valid)
            total_n = len(list(m.layer2_results))

            # Diagnosis distribution from layer1 results
            diag_counts: dict[str, int] = {}
            for r in list(m.layer1_results):
                d = getattr(r, "diagnosis", "unknown")
                diag_counts[d] = diag_counts.get(d, 0) + 1

            print()
            print(f"{'═' * 76}")
            print(f"  Run ID: {run_id}    Mode: {self._eval_mode}    {self.total_queries} queries    耗时: {self._format_time(elapsed)}")
            print(f"{'═' * 76}")
            print()
            print("  Layer 1                                             vs 上次")
            print("  ────────────")
            for name, key in [("Recall@5", "recall_at_k"), ("Precision@5", "precision_at_k"),
                              ("Hit@5", "hit_at_k"), ("MRR", "mrr"),
                              ("MAP@5", "map_at_k"), ("NDCG@5", "ndcg_at_k")]:
                val = self._fmt_val(l1.get(key))
                delta = self._fmt_delta(deltas.get(key))
                print(f"  {name:<14s}  {val:>8s}                             {delta:>8s}")

            print()
            if diag_counts:
                diag_parts = "   ".join(f"{k}: {v}" for k, v in sorted(diag_counts.items()))
                print(f"  诊断分布:  {diag_parts}")

            print()
            delta_label = f"Δ (N={deltas.get('_matching', 0)} matching)" if deltas else ""
            print(f"  Layer 2 ({valid_n} valid / {total_n} total)                     vs 上次")
            print("  ────────────")
            for name, key in [("Faithfulness", "faithfulness"), ("Answer Relev", "answer_relevancy"),
                              ("Context Prec", "context_precision"), ("Context Rec", "context_recall"),
                              ("Answer Corr", "answer_correctness")]:
                val = self._fmt_val(l2.get(key))
                delta = self._fmt_delta(deltas.get(key))
                print(f"  {name:<14s}  {val:>8s}                             {delta:>8s}")
            if delta_label:
                print(f"  {' ' * 50}{delta_label}")

            print()
            v = verdicts
            print(f"  Verdict:  pass: {v.get('pass', 0)}   partial: {v.get('partial', 0)}"
                  f"   fail: {v.get('fail', 0)}   error: {v.get('error', 0)}")

            print()
            print("  阶段延迟 (ms)           P50     P75     P95")
            print("  ────────────────")
            for label, key in [("Retrieve", "retrieve"), ("Generate", "generate"),
                               ("Judge Faithfulness", "judge_faithfulness"),
                               ("Judge Quality", "judge_quality")]:
                s = stages.get(key, {})
                p50 = int(s.get("p50", 0))
                p75 = int(s.get("p75", 0))
                p95 = int(s.get("p95", 0))
                print(f"  {label:<20s}  {p50:>6d}  {p75:>6d}  {p95:>6d}")

            e2e = stages.get("end_to_end", {})
            print(f"  {'End-to-end':<20s}  {int(e2e.get('p50', 0)):>6d}  {int(e2e.get('p75', 0)):>6d}  {int(e2e.get('p95', 0)):>6d}")

            print()
            print("  LLM 调用统计")
            print("  ────────────")
            print(f"  Generator: {m.generator_llm_calls} 次  |  "
                  f"Judge Faithfulness: {m.judge_faithfulness_calls} 次  |  "
                  f"Judge Quality: {m.judge_quality_calls} 次")
            print(f"  Token: {m.total_input_tokens:,} in / {m.total_output_tokens:,} out")
            print(f"  Input P50/P95:  {int(m._p95([m.total_input_tokens // max(1, m.total_llm_calls)])) if m.total_llm_calls > 0 else 0}"
                  f" / {int(m.input_tokens_p95)} tok  |  "
                  f"Output P50/P95: {int(m._p95([m.total_output_tokens // max(1, m.total_llm_calls)])) if m.total_llm_calls > 0 else 0}"
                  f" / {int(m.output_tokens_p95)} tok")
            print(f"  重试: {m.retry_count} 次  |  解析失败: {m.parse_error_count} 次")

            print()
            print("  缓存 / 成本")
            print("  ────────────")
            gen_rate = f"{m.generator_cache_hits} hit / {m.generator_cache_misses} miss ({m.generator_cache_hit_rate:.1%})"
            judge_rate = f"{m.judge_cache_hits} hit / {m.judge_cache_misses} miss ({m.judge_cache_hit_rate:.1%})"
            print(f"  Generator 缓存: {gen_rate}")
            print(f"  Judge 缓存:     {judge_rate}")
            print(f"  估算成本:       ¥{m.estimated_cost_usd:.2f}")

            print()
            print("  错误明细")
            print("  ────────────")
            if m.error_types:
                for err_type, count in sorted(m.error_types.items(), key=lambda x: -x[1]):
                    print(f"  {err_type}: {count}")
            else:
                print("  (无错误)")

            print()
            print(f"{'═' * 76}")
        except Exception:
            print("\n(最终报告生成失败)")

    # ============================================================
    # Helpers
    # ============================================================

    def _compute_deltas(self, metrics: "MonitorMetrics") -> dict:
        """对已完成的 query 计算与上次运行的差值均值。"""
        if not self.previous_per_query:
            return {}
        deltas_accum: dict[str, list[float]] = {}
        matching = 0
        for r in list(metrics.layer2_results):
            qid = getattr(r, "query_id", None)
            if qid not in self.previous_per_query:
                continue
            prev = self.previous_per_query[qid]
            matching += 1
            for field in ["faithfulness", "answer_relevancy", "context_precision",
                          "context_recall", "answer_correctness"]:
                cur_val = getattr(r, field, None)
                prev_val = prev.get(field)
                if cur_val is not None and prev_val is not None:
                    deltas_accum.setdefault(field, []).append(cur_val - prev_val)

        # Also compute L1 deltas from layer1_results
        l1_deltas: dict[str, list[float]] = {}
        l1_prev_count = 0
        for r in list(metrics.layer1_results):
            qid = getattr(r, "query_id", None)
            if not qid or qid not in self.previous_per_query:
                continue
            prev = self.previous_per_query[qid]
            l1_prev_count += 1
            for field in ["recall_at_k", "precision_at_k", "hit_at_k", "mrr", "map_at_k", "ndcg_at_k"]:
                cur_val = getattr(r, field, None)
                prev_val = prev.get(field)
                if cur_val is not None and prev_val is not None:
                    l1_deltas.setdefault(field, []).append(cur_val - prev_val)

        result = {}
        for f, vals in {**deltas_accum, **l1_deltas}.items():
            if vals:
                result[f] = sum(vals) / len(vals)
        result["_matching"] = matching
        return result

    @staticmethod
    def _format_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    @staticmethod
    def _fmt_val(value) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    @staticmethod
    def _fmt_delta(value) -> str:
        if value is None:
            return "—"
        if isinstance(value, (int, float)):
            sign = "+" if value >= 0 else "−"
            return f"{sign}{abs(value):.4f}"
        return str(value)

    def _terminal_reset(self):
        if self._mode == "ansi":
            sys.stdout.write(C_RESET + CURSOR_SHOW)
            sys.stdout.flush()


# ============================================================
# Module-level singleton
# ============================================================

_panel: LivePanel | None = None


def get_panel() -> LivePanel | None:
    """模块级单例。未初始化时返回 None，调用方自行判空。"""
    return _panel


def set_panel(panel: LivePanel) -> None:
    global _panel
    _panel = panel
