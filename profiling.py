"""Run instrumentation: per-stage wall-clock time, call counts, tokens, memory.

A ``Profiler`` accumulates named stages across a whole run, so a 20-county sweep
reports where the time actually went rather than one total. Usage:

    prof = Profiler()
    with prof.stage("geometric_score", county="FEJER"):
        ...
    prof.count("kernel_fits", 54)
    prof.report()          # printed table
    prof.save(path)        # JSON, one record per stage entry

Memory is reported two ways, because the two processes that matter are
separate. ``ru_maxrss`` is the peak resident size of this Python process, and
``sample_ollama_rss`` reads the resident size of the local model server, which
holds the weights and is by far the larger of the two.
"""

from __future__ import annotations

import json
import platform
import resource
import subprocess
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path


def peak_rss_gb():
    """Peak resident size of this process. macOS reports bytes, Linux kibibytes."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1073741824 if platform.system() == "Darwin" else raw / 1048576


def sample_ollama_rss_gb():
    """Resident size of the local model server, or None when it is not running."""
    try:
        pids = subprocess.run(
            ["pgrep", "-f", "llama-server"], capture_output=True, text=True, timeout=5
        ).stdout.split()
        if not pids:
            return None
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", ",".join(pids)],
            capture_output=True, text=True, timeout=5,
        ).stdout.split()
        return sum(int(v) for v in out) / 1048576
    except Exception:
        return None


class Profiler:
    def __init__(self):
        self.started = time.time()
        self.records = []
        self.totals = defaultdict(float)
        self.calls = defaultdict(int)
        self.counters = defaultdict(int)
        self.peak_ollama_rss = 0.0

    @contextmanager
    def stage(self, name, **meta):
        t0 = time.time()
        try:
            yield
        finally:
            dt = time.time() - t0
            self.totals[name] += dt
            self.calls[name] += 1
            self.records.append({"stage": name, "seconds": round(dt, 3), **meta})

    def count(self, name, n=1):
        self.counters[name] += n

    def note_memory(self):
        rss = sample_ollama_rss_gb()
        if rss:
            self.peak_ollama_rss = max(self.peak_ollama_rss, rss)
        return rss

    # ------------------------------------------------------------------
    def summary(self, llm_stats=None):
        elapsed = time.time() - self.started
        accounted = sum(self.totals.values())
        return {
            "wall_clock_seconds": round(elapsed, 1),
            "accounted_seconds": round(accounted, 1),
            "unaccounted_seconds": round(elapsed - accounted, 1),
            "stages": {
                name: {
                    "seconds": round(total, 1),
                    "share_pct": round(100 * total / elapsed, 1) if elapsed else 0.0,
                    "entries": self.calls[name],
                    "mean_seconds": round(total / self.calls[name], 3),
                }
                for name, total in sorted(
                    self.totals.items(), key=lambda kv: -kv[1]
                )
            },
            "counters": dict(self.counters),
            "llm": llm_stats or {},
            "memory_gb": {
                "python_peak_rss": round(peak_rss_gb(), 2),
                "model_server_peak_rss": round(self.peak_ollama_rss, 2) or None,
            },
        }

    def report(self, llm_stats=None):
        s = self.summary(llm_stats)
        print("\n" + "=" * 72)
        print("PROFILE")
        print("=" * 72)
        print(f"wall clock {s['wall_clock_seconds']:.0f}s "
              f"({s['wall_clock_seconds']/60:.1f} min), "
              f"accounted {s['accounted_seconds']:.0f}s, "
              f"unaccounted {s['unaccounted_seconds']:.0f}s")
        print(f"\n{'stage':<22}{'seconds':>10}{'share':>8}{'entries':>9}{'mean s':>10}")
        print("-" * 72)
        for name, st in s["stages"].items():
            print(f"{name:<22}{st['seconds']:>10.1f}{st['share_pct']:>7.1f}%"
                  f"{st['entries']:>9}{st['mean_seconds']:>10.3f}")
        if s["counters"]:
            print("\ncounters")
            for k, v in sorted(s["counters"].items()):
                print(f"  {k:<28}{v:>10,}")
        if s["llm"]:
            print("\nLLM")
            for k, v in s["llm"].items():
                print(f"  {k:<28}{v:>10,}" if isinstance(v, int)
                      else f"  {k:<28}{v:>10}")
        print("\nmemory, GB")
        for k, v in s["memory_gb"].items():
            print(f"  {k:<28}{v}")
        return s

    def save(self, path, llm_stats=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"summary": self.summary(llm_stats), "records": self.records}, indent=2
        ))
        print(f"\nProfile written to {path}")
