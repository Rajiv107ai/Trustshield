#!/usr/bin/env python3
"""TrustShield AI — Durable Action Command Outbox Background Worker Daemon.

Consumes pending operator action commands from the durable outbox (PostgreSQL
in production, SQLite in development) and executes them against the appropriate
provider adapters outside the synchronous request-response path.

Usage:
    # Run continuous polling daemon
    python scripts/run_action_worker.py --poll-interval 2.0 --batch-size 10

    # Process pending batch once and exit (for testing / cron)
    python scripts/run_action_worker.py --once
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.action_outbox import (  # noqa: E402
    ActionOutbox,
    ActionOutboxConfigurationError,
)

logger = logging.getLogger("trustshield.action_worker")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class ActionWorkerDaemon:
    """Manages the lifecycle and execution loop of the action outbox worker."""

    def __init__(
        self,
        outbox: Optional[ActionOutbox] = None,
        poll_interval: float = 2.0,
        batch_size: int = 10,
        max_iterations: Optional[int] = None,
    ) -> None:
        self._owns_outbox = outbox is None
        self.outbox = outbox or ActionOutbox()
        self.poll_interval = max(0.1, poll_interval)
        self.batch_size = max(1, batch_size)
        self.max_iterations = max_iterations
        self._running = False
        self._stop_requested = False

    def request_stop(self, signum: int = 0, frame: object = None) -> None:
        logger.info("Termination signal received (%s). Shutting down action worker cleanly...", signum)
        self._stop_requested = True

    def run(self, once: bool = False) -> int:
        self._running = True
        self._stop_requested = False

        # Register graceful signal handlers if supported
        try:
            signal.signal(signal.SIGINT, self.request_stop)
            signal.signal(signal.SIGTERM, self.request_stop)
        except (ValueError, AttributeError):
            pass

        config = self.outbox.configuration
        logger.info(
            "Action worker started [backend=%s, env=%s, batch_size=%d, interval=%.1fs]",
            config.backend,
            config.environment,
            self.batch_size,
            self.poll_interval,
        )

        processed_total = 0
        iteration = 0

        try:
            while not self._stop_requested:
                iteration += 1
                try:
                    receipts = self.outbox.process_pending(limit=self.batch_size)
                    if receipts:
                        processed_total += len(receipts)
                        for receipt in receipts:
                            logger.info(
                                "Processed command %s (action=%s, status=%s, attempts=%d)",
                                receipt.command_id,
                                receipt.action.value,
                                receipt.status.value,
                                receipt.attempts,
                            )
                except Exception as exc:
                    logger.error("Error processing pending action commands: %s", exc, exc_info=True)

                if once:
                    break

                if self.max_iterations is not None and iteration >= self.max_iterations:
                    logger.info("Reached maximum iterations (%d). Stopping worker.", self.max_iterations)
                    break

                # Sleep in short increments to remain responsive to SIGINT/SIGTERM
                stop_time = time.time() + self.poll_interval
                while not self._stop_requested and time.time() < stop_time:
                    time.sleep(min(0.2, self.poll_interval))

        finally:
            self._running = False
            self.close()
            logger.info("Action worker stopped. Total commands processed: %d", processed_total)

        return 0

    def close(self) -> None:
        if self._owns_outbox:
            try:
                if hasattr(self.outbox, "close"):
                    self.outbox.close()
            except Exception as exc:
                logger.warning("Error closing action outbox: %s", exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TrustShield AI — Action Command Outbox Background Worker Daemon"
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Polling interval in seconds between checks for pending commands (default: 2.0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Maximum number of commands to process per batch (default: 10)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process a single batch of pending commands and exit immediately",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum number of polling iterations before stopping (default: unlimited)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose debug logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(verbose=args.verbose)

    try:
        worker = ActionWorkerDaemon(
            poll_interval=args.poll_interval,
            batch_size=args.batch_size,
            max_iterations=args.max_iterations,
        )
    except ActionOutboxConfigurationError as exc:
        logger.critical("Failed to initialize ActionOutbox: %s", exc)
        return 1

    return worker.run(once=args.once)


if __name__ == "__main__":
    sys.exit(main())
