import yaml


def load(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def steps_per_epoch(cfg: dict, shard_len: int) -> int:
    return shard_len // cfg["batch_size"]
