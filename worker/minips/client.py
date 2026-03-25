import grpc
import numpy as np

from .pb import ps_pb2, ps_pb2_grpc

CHUNK_BYTES = 4 << 20
MAX_MSG = 64 << 20


class PSClient:
    def __init__(self, addr: str, worker_id: int):
        self.worker_id = worker_id
        self.channel = grpc.insecure_channel(
            addr,
            options=[("grpc.max_receive_message_length", MAX_MSG), ("grpc.max_send_message_length", MAX_MSG)],
        )
        self.stub = ps_pb2_grpc.ParameterServerStub(self.channel)
        self.md = (("worker-id", str(worker_id)),)
        self.specs = []

    def wait_ready(self, timeout: float = 30.0):
        grpc.channel_ready_future(self.channel).result(timeout=timeout)

    def register(self):
        reply = self.stub.Register(ps_pb2.RegisterRequest(worker_id=self.worker_id), metadata=self.md)
        self.specs = list(reply.params)
        return reply

    def pull(self, clock: int) -> tuple[dict[str, np.ndarray], int]:
        bufs = {s.name: bytearray(4 * int(np.prod(s.shape, dtype=np.int64))) for s in self.specs}
        min_clock = 0
        req = ps_pb2.PullRequest(worker_id=self.worker_id, clock=clock)
        for chunk in self.stub.Pull(req, metadata=self.md):
            bufs[chunk.name][chunk.offset : chunk.offset + len(chunk.data)] = chunk.data
            min_clock = chunk.min_clock
        return {n: np.frombuffer(b, "<f4") for n, b in bufs.items()}, min_clock

    def push(self, clock: int, grads: list[tuple[str, int, bytes]]):
        def chunks():
            for name, enc, data in grads:
                for off in range(0, max(len(data), 1), CHUNK_BYTES):
                    yield ps_pb2.GradChunk(
                        worker_id=self.worker_id, clock=clock, name=name, encoding=enc,
                        data=data[off : off + CHUNK_BYTES],
                    )

        self.stub.Push(chunks(), metadata=self.md)

    def clock(self, clock: int):
        self.stub.Clock(ps_pb2.ClockRequest(worker_id=self.worker_id, clock=clock), metadata=self.md)

    def stats(self) -> dict:
        r = self.stub.Stats(ps_pb2.StatsRequest(), metadata=self.md)
        return {
            "workers": [
                {
                    "worker_id": w.worker_id, "bytes_in": w.bytes_in, "bytes_out": w.bytes_out,
                    "wait_seconds": w.wait_seconds, "pulls": w.pulls, "pushes": w.pushes,
                }
                for w in r.workers
            ],
            "staleness_hist": {int(k): int(v) for k, v in sorted(r.staleness_hist.items())},
            "updates_applied": r.updates_applied,
        }

    def close(self):
        self.channel.close()
