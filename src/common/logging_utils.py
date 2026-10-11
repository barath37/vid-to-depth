import time
import logging
from contextlib import contextmanager
from typing import Dict


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("vid2depth")


class StageTimer:
    """Tracks cumulative and stage-by-stage wall-clock durations against the 15-minute budget."""

    def __init__(self, budget_minutes: float = 15.0):
        self.budget_seconds = budget_minutes * 60.0
        self.stage_timings: Dict[str, float] = {}
        self.start_time = time.time()

    @contextmanager
    def measure(self, stage_name: str):
        t0 = time.time()
        logger.info(f"==> Starting Stage: {stage_name}")
        try:
            yield
        finally:
            dt = time.time() - t0
            self.stage_timings[stage_name] = dt
            elapsed_total = time.time() - self.start_time
            remaining = self.budget_seconds - elapsed_total
            logger.info(
                f"<== Completed Stage: {stage_name} in {dt:.2f}s "
                f"(Total elapsed: {elapsed_total:.2f}s, Budget remaining: {remaining:.2f}s)"
            )

    def summary(self) -> Dict[str, float]:
        return {
            **self.stage_timings,
            "total_elapsed_seconds": time.time() - self.start_time,
            "within_budget": (time.time() - self.start_time) <= self.budget_seconds
        }
