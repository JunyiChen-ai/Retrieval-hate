"""Formal static policy for new dense assets; old K30 aggregates are retired."""
from pathlib import Path
from .common import sha256_file

DENIED_OLD_K30={
 "8b4a706cec51d106151e57109b24850232239168d5e0ca363341ee76493d7fb7",
 "a2ae105e61478b86193267fe67263d1c26436f0881620222f0aa1544fa380778",
}
DENIED_NAME_TOKEN="subclipK30_openai_clip-vit-large-patch14-336_HF.pt"

def assert_new_dense_asset(path:Path):
    if DENIED_NAME_TOKEN in path.name or (path.exists() and sha256_file(path) in DENIED_OLD_K30):
        raise RuntimeError("HALT_RETIRED_OLD_K30_CACHE")
    return path

def empty_dense_status(frames):
    if len(frames)!=4:raise RuntimeError("HALT_DENSE_FRAME_COUNT")
    return "OUTCOME" if any(x.get("decode_status")=="ok" for x in frames) else "EMPTY_DENSE"
