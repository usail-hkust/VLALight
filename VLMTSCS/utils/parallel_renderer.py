"""Isolated-process batch rendering for the VLM video collector.

The normal collector owns one :class:`TSHubRenderer` in the parent process.
Panda3D/EGL contexts are not safe to call concurrently, so the parallel path
uses one renderer process per configured worker.  Each worker can switch
between its assigned sensor batches.  The parent only sends plain SUMO
observations to those workers and performs image/video persistence itself.

This module is intentionally imported lazily by ``VLMOneLine``.  Deployments
that keep the original serial renderer do not create multiprocessing objects or
import TransSimHub through this module.
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import queue
import traceback
from collections.abc import Mapping, Sequence
from typing import Any


_RENDERER_IMPORT = (
    "TransSimHub.tshub.tshub_env3d.vis3d_renderer.tshub_render"
)


def _subset_tls_init_info(tls_init_info: Any, tls_ids: Sequence[str]) -> Any:
    """Keep only the assigned TLS entries in a renderer reset payload."""
    ids = {str(value) for value in tls_ids}
    if not isinstance(tls_init_info, Mapping):
        return tls_init_info

    # Current SUMO integration returns {"tls": {tls_id: ...}, ...}.
    tls_payload = tls_init_info.get("tls")
    if isinstance(tls_payload, Mapping):
        result = dict(tls_init_info)
        result["tls"] = {
            key: value for key, value in tls_payload.items() if str(key) in ids
        }
        return result

    # Keep compatibility with a direct {tls_id: info} mapping used by older
    # TransSimHub adapters.
    if all(str(key) in ids for key in tls_init_info):
        return {
            key: value for key, value in tls_init_info.items() if str(key) in ids
        }
    return tls_init_info


def _worker_main(
    task_queue: Any,
    result_queue: Any,
    renderer_kwargs: dict[str, Any],
    tls_init_info: Any,
    worker_index: int,
) -> None:
    """Create one isolated renderer and serve render requests."""
    renderer = None
    try:
        module = __import__(_RENDERER_IMPORT, fromlist=["TSHubRenderer"])
        renderer_cls = module.TSHubRenderer
        renderer = renderer_cls(**renderer_kwargs)
        renderer.reset(tls_init_info)
        result_queue.put(("ready", int(worker_index), -1, None))

        while True:
            message = task_queue.get()
            if message is None or message[0] == "close":
                break
            (
                task_id,
                local_batch_index,
                tshub_obs,
                should_count_vehicles,
            ) = message
            if hasattr(renderer, "switch_to_batch"):
                renderer.switch_to_batch(int(local_batch_index))
            sensor_data = renderer.step(
                tshub_obs,
                should_count_vehicles=bool(should_count_vehicles),
            )
            result_queue.put(("result", int(worker_index), int(task_id), sensor_data))
    except BaseException as exc:  # propagate child failures to the parent
        result_queue.put(
            (
                "error",
                int(worker_index),
                -1,
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            )
        )
    finally:
        if renderer is not None:
            try:
                renderer.destroy()
            except BaseException:
                pass


class ParallelTSHubRenderer:
    """Process-isolated renderer facade used by the VLM collector.

    A call to :meth:`step_all` submits one task for each batch.  At most
    ``workers`` renderer processes are active, and each process reuses one
    isolated renderer context for the batches assigned to it.
    """

    is_parallel = True

    def __init__(
        self,
        *,
        tls_ids: Sequence[str],
        tls_init_info: Any,
        batch_size: int | None,
        workers: int,
        renderer_kwargs: Mapping[str, Any],
        timeout_s: float = 600.0,
    ) -> None:
        self.tls_ids = [str(value) for value in tls_ids]
        if not self.tls_ids:
            raise ValueError("parallel rendering requires at least one TLS")
        if batch_size is None or int(batch_size) <= 0:
            batch_size = len(self.tls_ids)
        self.batch_size = int(batch_size)
        self.batches = [
            self.tls_ids[start : start + self.batch_size]
            for start in range(0, len(self.tls_ids), self.batch_size)
        ]
        self.workers = max(1, min(int(workers), len(self.batches)))
        self.worker_batch_indices = [
            [] for _ in range(self.workers)
        ]
        for batch_index in range(len(self.batches)):
            self.worker_batch_indices[batch_index % self.workers].append(batch_index)
        self.batch_to_worker = {
            batch_index: worker_index
            for worker_index, batch_indices in enumerate(self.worker_batch_indices)
            for batch_index in batch_indices
        }
        self.batch_to_local_index = {
            batch_index: local_index
            for batch_indices in self.worker_batch_indices
            for local_index, batch_index in enumerate(batch_indices)
        }
        self.timeout_s = max(1.0, float(timeout_s))
        self._context = mp.get_context("spawn")
        self._result_queue = self._context.Queue()
        self._task_queues: list[Any] = []
        self._processes: list[Any] = []
        self._next_task_id = 0
        self._closed = False

        try:
            for worker_index, worker_batch_indices in enumerate(self.worker_batch_indices):
                task_queue = self._context.Queue(maxsize=1)
                worker_kwargs = copy.deepcopy(dict(renderer_kwargs))
                sensor_config = copy.deepcopy(worker_kwargs.get("sensor_config", {}))
                worker_tls_ids = [
                    tls_id
                    for batch_index in worker_batch_indices
                    for tls_id in self.batches[batch_index]
                ]
                worker_tls_id_set = set(worker_tls_ids)
                if isinstance(sensor_config, Mapping):
                    sensor_config = dict(sensor_config)
                    all_tls = sensor_config.get("tls", {})
                    if isinstance(all_tls, Mapping):
                        sensor_config["tls"] = {
                            key: value
                            for key, value in all_tls.items()
                            if str(key) in worker_tls_id_set
                        }
                worker_kwargs["sensor_config"] = sensor_config
                worker_kwargs["tls_batch_size"] = (
                    self.batch_size if len(worker_batch_indices) > 1 else None
                )
                worker_kwargs["simid"] = f"sumo_parallel_worker_{worker_index}"
                process = self._context.Process(
                    target=_worker_main,
                    args=(
                        task_queue,
                        self._result_queue,
                        worker_kwargs,
                        _subset_tls_init_info(tls_init_info, worker_tls_ids),
                        worker_index,
                    ),
                    name=f"vlm-render-worker-{worker_index}",
                    daemon=True,
                )
                process.start()
                self._task_queues.append(task_queue)
                self._processes.append(process)

            self._wait_until_ready()
        except BaseException:
            self.close()
            raise

    def get_batch_info(self) -> dict[str, Any]:
        return {
            "mode": "parallel",
            "batch_size": self.batch_size,
            "total_batches": len(self.batches),
            "workers": self.workers,
        }

    def get_all_batch_tls_ids(self) -> list[list[str]]:
        return [list(batch) for batch in self.batches]

    def _raise_if_worker_died(self, context: str) -> None:
        dead = [
            (index, process.exitcode)
            for index, process in enumerate(self._processes)
            if not process.is_alive()
        ]
        if not dead:
            return
        self.close()
        details = ", ".join(
            f"worker={index}, exitcode={exitcode}"
            for index, exitcode in dead
        )
        raise RuntimeError(
            f"parallel render worker exited unexpectedly while {context}: {details}"
        )

    def _wait_until_ready(self) -> None:
        pending = set(range(self.workers))
        while pending:
            try:
                kind, worker_index, _task_id, payload = self._result_queue.get(
                    timeout=self.timeout_s
                )
            except queue.Empty as exc:
                self._raise_if_worker_died("starting workers")
                self.close()
                raise RuntimeError("timed out while starting parallel render workers") from exc
            if kind == "ready":
                pending.discard(int(worker_index))
            elif kind == "error":
                self.close()
                raise RuntimeError(
                    f"parallel render worker {worker_index} failed during startup:\n{payload}"
                )

    def step_all(
        self,
        batch_inputs: Sequence[tuple[Sequence[str], Any]],
        *,
        should_count_vehicles: bool = False,
    ) -> list[Any]:
        """Render all batches and return sensor data in batch order."""
        if self._closed:
            raise RuntimeError("parallel renderer is already closed")
        if len(batch_inputs) != len(self.batches):
            raise ValueError(
                f"expected {len(self.batches)} batch inputs, got {len(batch_inputs)}"
            )

        outputs: list[Any] = [None] * len(self.batches)
        for wave_start in range(0, len(self.batches), self.workers):
            wave_indices = list(
                range(wave_start, min(wave_start + self.workers, len(self.batches)))
            )
            pending: dict[int, int] = {}
            for batch_index in wave_indices:
                _batch_tls_ids, tshub_obs = batch_inputs[batch_index]
                task_id = self._next_task_id
                self._next_task_id += 1
                pending[task_id] = batch_index
                worker_index = self.batch_to_worker[batch_index]
                local_batch_index = self.batch_to_local_index[batch_index]
                try:
                    self._task_queues[worker_index].put(
                        (
                            task_id,
                            local_batch_index,
                            tshub_obs,
                            bool(should_count_vehicles),
                        ),
                        timeout=self.timeout_s,
                    )
                except queue.Full as exc:
                    self.close()
                    raise RuntimeError(
                        f"timed out submitting batch {batch_index} to render worker"
                    ) from exc

            while pending:
                try:
                    kind, worker_index, task_id, payload = self._result_queue.get(
                        timeout=self.timeout_s
                    )
                except queue.Empty as exc:
                    self._raise_if_worker_died("rendering a batch wave")
                    self.close()
                    raise RuntimeError(
                        "timed out while waiting for parallel render workers"
                    ) from exc
                if kind == "error":
                    self.close()
                    raise RuntimeError(
                        f"parallel render worker {worker_index} failed:\n{payload}"
                    )
                if kind != "result" or int(task_id) not in pending:
                    continue
                outputs[pending.pop(int(task_id))] = payload
        return outputs

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for task_queue in self._task_queues:
            try:
                task_queue.put_nowait(("close",))
            except Exception:
                pass
        for process in self._processes:
            try:
                process.join(timeout=5.0)
            except Exception:
                pass
        for process in self._processes:
            if process.is_alive():
                try:
                    process.terminate()
                    process.join(timeout=2.0)
                except Exception:
                    pass
        for task_queue in self._task_queues:
            try:
                task_queue.close()
            except Exception:
                pass
        try:
            self._result_queue.close()
        except Exception:
            pass

    destroy = close

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown path
        try:
            self.close()
        except Exception:
            pass
