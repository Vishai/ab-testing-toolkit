"""Copy a finished run folder to S3 so dashboards and people can read it."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"not an s3:// URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/").rstrip("/")


def publish_run(run_dir: Path | str, uri: str, client=None) -> list[str]:
    """Upload every file in the run folder under s3://bucket/prefix/<run_id>/.

    The experiment's raw logs stay out of it; only results, the manifest and the
    dashboard bundle are published.
    """
    bucket, prefix = parse_s3_uri(uri)
    if client is None:
        import boto3

        client = boto3.client("s3")
    folder = Path(run_dir)
    keys = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or "experiment" in path.relative_to(folder).parts:
            continue
        key = "/".join(filter(None, [prefix, folder.name, path.relative_to(folder).as_posix()]))
        client.upload_file(str(path), bucket, key)
        keys.append(key)
    return keys
