"""Backblaze B2 artifact push/pull helpers for the RGCL-video project.

Thin wrapper around the `rclone` CLI. Importing this module has no side
effects; all work happens inside the functions / the ``__main__`` CLI.

Base prefix:
    b2:junyi-data/RGCL_video

Examples
--------
    from utils.b2_artifacts import push, pull, exists
    push("data/CLIP_Embedding/MHC", "embeddings/MHC")
    pull("embeddings/MHC", "data/CLIP_Embedding/MHC")
    exists("embeddings/MHC")          # -> bool

CLI:
    python b2_artifacts.py push <local> <subpath> [--move]
    python b2_artifacts.py pull <subpath> <local>
    python b2_artifacts.py exists <subpath>
"""

import argparse
import subprocess
import sys

B2_BASE = "b2:junyi-data/RGCL_video"


def _dest(b2_subpath):
    """Join the base prefix with a sub-path under RGCL_video."""
    return "{}/{}".format(B2_BASE, b2_subpath.lstrip("/"))


def push(local_path, b2_subpath, move=False):
    """Upload ``local_path`` to ``B2_BASE/b2_subpath``.

    Uses ``rclone move`` when ``move=True`` (deletes source on success),
    otherwise ``rclone copy``.
    """
    dest = _dest(b2_subpath)
    cmd = "move" if move else "copy"
    args = [
        "rclone", cmd, local_path, dest,
        "--transfers", "8", "--b2-hard-delete",
    ]
    print("[b2_artifacts] {}: {} -> {}".format(cmd, local_path, dest))
    subprocess.run(args, check=True)


def pull(b2_subpath, local_path):
    """Download ``B2_BASE/b2_subpath`` into ``local_path`` (creates parent)."""
    import os

    src = _dest(b2_subpath)
    parent = os.path.dirname(local_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    args = [
        "rclone", "copy", src, local_path,
        "--transfers", "8",
    ]
    print("[b2_artifacts] copy: {} -> {}".format(src, local_path))
    subprocess.run(args, check=True)


def exists(b2_subpath):
    """Return True if anything is listed at ``B2_BASE/b2_subpath``."""
    target = _dest(b2_subpath)
    result = subprocess.run(
        ["rclone", "lsf", target],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return False
    return bool(result.stdout.strip())


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Push/pull RGCL-video artifacts to/from Backblaze B2 via rclone.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_push = sub.add_parser("push", help="upload a local path to B2")
    p_push.add_argument("local", help="local file or directory")
    p_push.add_argument("subpath", help="destination sub-path under RGCL_video")
    p_push.add_argument(
        "--move", action="store_true",
        help="use rclone move (delete source on success) instead of copy",
    )

    p_pull = sub.add_parser("pull", help="download a B2 path to local")
    p_pull.add_argument("subpath", help="source sub-path under RGCL_video")
    p_pull.add_argument("local", help="local destination file or directory")

    p_exists = sub.add_parser("exists", help="check whether a B2 sub-path exists")
    p_exists.add_argument("subpath", help="sub-path under RGCL_video to check")

    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.command == "push":
        push(args.local, args.subpath, move=args.move)
    elif args.command == "pull":
        pull(args.subpath, args.local)
    elif args.command == "exists":
        print(exists(args.subpath))
    else:  # pragma: no cover - argparse enforces a valid command
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
