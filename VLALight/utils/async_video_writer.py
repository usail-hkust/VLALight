from concurrent.futures import Future, ThreadPoolExecutor
from threading import BoundedSemaphore, Lock


class AsyncVideoWritePool:
    def __init__(self, num_workers=4, max_queue_size=None):
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(num_workers)))
        queue_limit = int(max_queue_size or 0)
        self._pending_slots = BoundedSemaphore(queue_limit) if queue_limit > 0 else None
        self._shutdown = False

    def submit(self, fn, *args, **kwargs):
        if self._shutdown:
            raise RuntimeError("AsyncVideoWritePool has been shut down.")
        if self._pending_slots is not None:
            self._pending_slots.acquire()
        try:
            future = self._executor.submit(fn, *args, **kwargs)
        except BaseException:
            if self._pending_slots is not None:
                self._pending_slots.release()
            raise
        if self._pending_slots is not None:
            future.add_done_callback(lambda _: self._pending_slots.release())
        return future

    def shutdown(self):
        self._shutdown = True
        self._executor.shutdown(wait=True)


class AsyncVideoWriter:
    def __init__(self, writer, pool: AsyncVideoWritePool, copy_frame=True):
        self._writer = writer
        self._pool = pool
        self._lock = Lock()
        self._tail = Future()
        self._tail.set_result(None)
        self._released = False
        self._copy_frame = copy_frame

    def _write_after(self, previous_future, frame):
        previous_future.result()
        with self._lock:
            self._writer.write(frame)

    def write(self, frame):
        if self._released:
            raise RuntimeError("Cannot write to a released AsyncVideoWriter.")
        queued_frame = frame.copy() if self._copy_frame else frame
        self._tail = self._pool.submit(self._write_after, self._tail, queued_frame)

    def release(self):
        if self._released:
            return
        self._released = True
        self._tail.result()
        with self._lock:
            self._writer.release()
