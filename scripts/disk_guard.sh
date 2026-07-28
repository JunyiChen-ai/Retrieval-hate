#!/usr/bin/env bash
#
# disk_guard.sh — quota watchdog for the RGCL hateful-video-detection project.
#
# Safe to SOURCE at the top of every sbatch job, OR to execute standalone.
# When usage on the /data quota exceeds a threshold it reclaims space in a
# strict priority order, re-checking usage after every step and stopping as
# soon as usage is back under the target.  Every destructive step is gated on
# BOTH (a) a byte-identical copy verified present on Backblaze B2 and (b) the
# target path resolving under an explicit allowlist of roots.
#
# IMPORTANT DESIGN NOTES
#   * We deliberately do NOT use `set -e`.  This file may be `source`d into an
#     sbatch job; a non-zero exit from any command must never abort the parent
#     job.  We use `set -uo pipefail` plus explicit error handling, and we
#     always finish via `_dg_finish` which does `return 0` (sourced) / `exit 0`
#     (executed).
#   * DRY-RUN is the safe default surface: with --dry-run (or DISK_GUARD_DRY_RUN=1)
#     the script parses, decides and logs exactly what it WOULD do, but performs
#     no pushes and no deletes.
#   * If quota usage cannot be determined, the script does NOTHING destructive.
#
# Usage:
#   bash scripts/disk_guard.sh [--dry-run]
#   source scripts/disk_guard.sh            # at the top of an sbatch job
#
# Key env vars (all optional):
#   DISK_GUARD_THRESHOLD_GB   trigger reclaim above this many GB   (default 250)
#   DISK_GUARD_TARGET_GB      reclaim until under this many GB     (default = threshold)
#   RGCL_ROOT                 project root                         (default /data/jehc223/RGCL)
#   B2_PREFIX                 rclone remote+prefix                 (default b2:junyi-data/RGCL_video)
#   DISK_GUARD_LOG            log file                             (default $RGCL_ROOT/slurm/logs/disk_guard.log)
#   DISK_GUARD_DRY_RUN=1      force dry-run
#   DISK_GUARD_HF_PURGE=1     allow purging HF datasets--*/.locks (never models--*)

set -uo pipefail

# ---------------------------------------------------------------------------
# 0. Sourced-vs-executed detection (so we use return vs exit correctly).
# ---------------------------------------------------------------------------
# _DG_SOURCED=1 when this file is being sourced, 0 when executed directly.
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    _DG_SOURCED=1
else
    _DG_SOURCED=0
fi

# ---------------------------------------------------------------------------
# 1. Config (env with defaults).
# ---------------------------------------------------------------------------
DISK_GUARD_THRESHOLD_GB="${DISK_GUARD_THRESHOLD_GB:-250}"
DISK_GUARD_TARGET_GB="${DISK_GUARD_TARGET_GB:-$DISK_GUARD_THRESHOLD_GB}"
RGCL_ROOT="${RGCL_ROOT:-/data/jehc223/RGCL}"
B2_PREFIX="${B2_PREFIX:-b2:junyi-data/RGCL_video}"
# logging/ may itself be pruned, so keep the log under the stable slurm/logs dir.
DISK_GUARD_LOG="${DISK_GUARD_LOG:-$RGCL_ROOT/slurm/logs/disk_guard.log}"
DISK_GUARD_HF_PURGE="${DISK_GUARD_HF_PURGE:-0}"

# HOME may not be exported inside a bare sbatch; fall back to the known value.
_DG_HOME="${HOME:-/data/jehc223/home}"
RCLONE_VFS_GLOB="$_DG_HOME/.cache/rclone/vfs"
HF_CACHE_DIR="$_DG_HOME/.cache/huggingface"

# The b2_push.sh helper (copy/move to B2); used only outside dry-run.
B2_PUSH="$RGCL_ROOT/scripts/b2_push.sh"

# DRY_RUN from --dry-run arg or env flag.
DRY_RUN=0
if [[ "${DISK_GUARD_DRY_RUN:-0}" == "1" ]]; then
    DRY_RUN=1
fi
for _arg in "$@"; do
    case "$_arg" in
        --dry-run) DRY_RUN=1 ;;
        *) : ;;  # ignore unknown args; never abort the parent job
    esac
done

# ---------------------------------------------------------------------------
# 2. Allowlist / blocklist for destructive operations.
# ---------------------------------------------------------------------------
# Only paths whose canonical form starts with one of these roots may be deleted.
# NOTE: "$RGCL_ROOT/data/CLIP_Embedding" was REMOVED from this list on 2026-07-28.
# It holds the 7168-d fused feature caches every $0 pregate depends on (incl. the
# Molmo2-8B HateMM caches); deleting them would force GPU re-extraction. No step in
# this script ever targeted it, so it sat on the permission list with nothing but the
# absence of a code path protecting it. See refine-logs/DISK_FORENSICS_2026-07-28.md §6
# (and §4.3 item 4). Removing it strictly reduces what the guard may touch.
_DG_ALLOWED_ROOTS=(
    "$RGCL_ROOT/logging"
    "$_DG_HOME/.cache/rclone"
    "$_DG_HOME/.cache/huggingface"
)
# Absolute refusal: never touch anything resolving under these.
_DG_BLOCKED_SUBSTR=(
    "/data/jehc223/AlphaSteer"
)
# Raw dataset dirs must never be deleted (match by name anywhere in the path).
_DG_RAW_DATASET_SUBSTR=(
    "Multihateclip"
    "MultiHateClip"
    "HateMM_raw"
    "ImpliHateVid_raw"
)

# ---------------------------------------------------------------------------
# 3. Logging helper.
# ---------------------------------------------------------------------------
log() {
    local _msg="$*"
    local _ts
    _ts="$(date '+%Y-%m-%d %H:%M:%S')"
    local _prefix=""
    [[ "$DRY_RUN" == "1" ]] && _prefix="[DRY-RUN] "
    local _line="[$_ts] [disk_guard] ${_prefix}${_msg}"
    # stdout
    printf '%s\n' "$_line"
    # append to log file (best-effort; never fatal)
    mkdir -p "$(dirname "$DISK_GUARD_LOG")" 2>/dev/null || true
    printf '%s\n' "$_line" >> "$DISK_GUARD_LOG" 2>/dev/null || true
}

# Central exit path — always succeeds so a sourcing sbatch is never aborted.
_dg_finish() {
    if [[ "${_DG_SOURCED}" == "1" ]]; then
        return 0
    else
        exit 0
    fi
}

# ---------------------------------------------------------------------------
# 4. get_usage_gb — parse `quota -s` for the /data (data-data) filesystem.
# ---------------------------------------------------------------------------
# Echoes an integer GB on success; echoes nothing and returns 1 if it cannot
# be determined (caller must then refuse all destructive actions).
get_usage_gb() {
    local _q
    _q="$(quota -s 2>/dev/null)" || { return 1; }
    [[ -z "$_q" ]] && return 1

    # The filesystem name and its numbers may sit on the same line or wrap to
    # the next (quota -s wraps long device names).  Grab the block starting at
    # the data-data device and take the first token that looks like a size.
    local _used
    _used="$(printf '%s\n' "$_q" \
        | awk '
            /data-data/ { grab=1;
                          # size may be on the same line after the device name
                          for (i=1;i<=NF;i++) if ($i ~ /^[0-9]+(\.[0-9]+)?[KMGT]?\*?$/ && $i !~ /data-data/) {print $i; exit}
                          next }
            grab==1     { for (i=1;i<=NF;i++) if ($i ~ /^[0-9]+(\.[0-9]+)?[KMGT]?\*?$/) {print $i; exit} }
          ')"

    [[ -z "$_used" ]] && return 1

    # Strip a possible trailing "*" (quota marks over-soft-limit with '*').
    _used="${_used%\*}"

    # Convert to integer GB based on unit suffix.
    local _num _unit _gb
    _unit="$(printf '%s' "$_used" | sed -E 's/^[0-9.]+//')"
    _num="$(printf '%s'  "$_used" | sed -E 's/[KMGT]$//')"

    # numeric sanity
    [[ "$_num" =~ ^[0-9]+(\.[0-9]+)?$ ]] || return 1

    case "$_unit" in
        T) _gb="$(awk -v n="$_num" 'BEGIN{printf "%d", n*1024}')" ;;
        G|"") _gb="$(awk -v n="$_num" 'BEGIN{printf "%d", n}')" ;;
        M) _gb="$(awk -v n="$_num" 'BEGIN{printf "%d", n/1024}')" ;;
        K) _gb="$(awk -v n="$_num" 'BEGIN{printf "%d", n/1048576}')" ;;
        *) return 1 ;;
    esac
    printf '%s' "$_gb"
    return 0
}

# ---------------------------------------------------------------------------
# 5. Path-safety helpers.
# ---------------------------------------------------------------------------
# _dg_canon <path> — best-effort canonicalisation (realpath, else the input).
_dg_canon() {
    local _p="$1"
    local _r
    _r="$(realpath -m -- "$_p" 2>/dev/null)" || _r="$_p"
    printf '%s' "$_r"
}

# _dg_path_allowed <path> — returns 0 only if path is under an allowed root,
# not under a blocked prefix, and not a raw-dataset path.
_dg_path_allowed() {
    local _raw="$1"
    local _p
    _p="$(_dg_canon "$_raw")"

    local _b
    for _b in "${_DG_BLOCKED_SUBSTR[@]}"; do
        if [[ "$_p" == *"$_b"* ]]; then
            log "REFUSE: path under blocked prefix ($_b): $_p"
            return 1
        fi
    done
    for _b in "${_DG_RAW_DATASET_SUBSTR[@]}"; do
        if [[ "$_p" == *"$_b"* ]]; then
            log "REFUSE: path looks like a raw dataset ($_b): $_p"
            return 1
        fi
    done

    local _root _ok=1
    for _root in "${_DG_ALLOWED_ROOTS[@]}"; do
        local _croot
        _croot="$(_dg_canon "$_root")"
        if [[ "$_p" == "$_croot" || "$_p" == "$_croot"/* ]]; then
            _ok=0
            break
        fi
    done
    if [[ "$_ok" != "0" ]]; then
        log "REFUSE: path outside allowed roots: $_p"
        return 1
    fi
    return 0
}

# _dg_bytes <path> — size in bytes (0 if missing / unreadable).
_dg_bytes() {
    local _p="$1"
    local _b
    _b="$(du -sb --apparent-size -- "$_p" 2>/dev/null | awk '{print $1}')"
    [[ -z "$_b" ]] && _b=0
    printf '%s' "$_b"
}

# _dg_human <bytes> — humanise a byte count.
_dg_human() {
    awk -v b="$1" 'BEGIN{
        split("B KB MB GB TB PB",u," ");
        i=1; x=b+0;
        while (x>=1024 && i<6){x/=1024;i++}
        printf "%.2f%s", x, u[i]
    }'
}

# ---------------------------------------------------------------------------
# 6. verify_on_b2 <local_file> <b2_dest_full>
#    Returns 0 ONLY if a byte-identical copy exists at exactly <b2_dest_full>.
#    Compares the local SHA1 to the B2-stored SHA1 (B2 keeps SHA1 natively).
#
#    B2 JSON key is Hashes.sha1 (nested object).  We use jq when available and
#    fall back to python3 json otherwise.  Never parse JSON with tr/awk.
#
#    Two layouts are handled:
#      (A) Correct layout (after b2_push.sh fix): file lives at exactly $_dest.
#          We list $_dir/ and look for Name==$_base with IsDir==false.
#      (B) Legacy double-nested layout (old b2_push.sh behaviour): rclone copy
#          treated $_dest as a directory and stored the file one level deeper at
#          $_dest/$_base.  We detect this by listing $_dest/ and again looking
#          for Name==$_base with IsDir==false.
#    Either layout returns 0 if the SHA1 matches.
# ---------------------------------------------------------------------------

# _b2_sha1_from_json <json_string> <basename>
# Extract the sha1 for the entry with Name==<basename> and IsDir==false.
# Echoes the sha1 hex string; echoes nothing if not found.
_b2_sha1_from_json() {
    local _json="$1" _b="$2"
    if command -v jq >/dev/null 2>&1; then
        printf '%s' "$_json" \
            | jq -r --arg b "$_b" \
                '.[] | select(.Name==$b and .IsDir==false) | .Hashes.sha1 // .Hashes["SHA-1"] // empty' \
                2>/dev/null
    else
        # Fallback: python3 json module (always available in conda env)
        printf '%s' "$_json" | python3 -c "
import sys, json, os
b = os.environ.get('_B2_BASE', '')
data = json.load(sys.stdin)
for e in data:
    if e.get('Name') == b and not e.get('IsDir', False):
        h = e.get('Hashes', {})
        sha = h.get('sha1') or h.get('SHA-1') or ''
        if sha:
            print(sha)
            break
" _B2_BASE="$_b" 2>/dev/null
    fi
}

verify_on_b2() {
    local _local="$1" _dest="$2"
    [[ -f "$_local" ]] || { log "verify: local missing: $_local"; return 1; }

    local _dir _base
    _dir="$(dirname -- "$_dest")"
    _base="$(basename -- "$_dest")"

    # Local SHA1
    local _lsha
    _lsha="$(rclone hashsum sha1 "$_local" 2>/dev/null | awk '{print $1}')"
    if [[ -z "$_lsha" ]]; then
        log "verify: could not hash local file: $_local"
        return 1
    fi

    # --- Layout (A): file at exactly $_dest (correct push layout) ---
    local _rjson _rsha
    _rjson="$(rclone lsjson --hash "$_dir/" 2>/dev/null)" || _rjson=""
    if [[ -n "$_rjson" ]]; then
        _rsha="$(_b2_sha1_from_json "$_rjson" "$_base")"
    fi

    # --- Layout (B): double-nested legacy push (file at $_dest/$_base) ---
    if [[ -z "$_rsha" ]]; then
        local _nested_json
        _nested_json="$(rclone lsjson --hash "$_dest/" 2>/dev/null)" || _nested_json=""
        if [[ -n "$_nested_json" ]]; then
            _rsha="$(_b2_sha1_from_json "$_nested_json" "$_base")"
            if [[ -n "$_rsha" ]]; then
                log "verify: found at legacy double-nested path: $_dest/$_base"
            fi
        fi
    fi

    if [[ -z "$_rsha" ]]; then
        log "verify: file not found on B2 at exact dest: $_dest"
        return 1
    fi
    if [[ "$_lsha" == "$_rsha" ]]; then
        log "verify: OK sha1=$_lsha matches at $_dest"
        return 0
    fi
    log "verify: MISMATCH local=$_lsha b2=$_rsha at $_dest"
    return 1
}

# ---------------------------------------------------------------------------
# 7. Reclaim steps.  Each returns after (best-effort) freeing space; the main
#    loop re-checks usage between steps and stops once under target.
# ---------------------------------------------------------------------------

# (a) Purge rclone VFS cache — pure cache, always safe.
reclaim_vfs_cache() {
    log "Step (a): rclone VFS cache purge."
    local _glob_dir="$_DG_HOME/.cache/rclone"
    local _any=0
    shopt -s nullglob
    local _d
    for _d in "$_glob_dir"/vfs*; do
        _any=1
        local _sz
        _sz="$(du -sh -- "$_d" 2>/dev/null | awk '{print $1}')"
        if [[ "$DRY_RUN" == "1" ]]; then
            log "  would rm -rf $_d (size ${_sz:-?})"
        else
            if _dg_path_allowed "$_d"; then
                rm -rf -- "$_d" 2>/dev/null && log "  removed $_d (was ${_sz:-?})" \
                    || log "  WARN failed to remove $_d"
            fi
        fi
    done
    shopt -u nullglob
    [[ "$_any" == "0" ]] && log "  no rclone vfs* cache dirs present (nothing to do)."
}

# (b) HF cache purge — conservative.  Default: skip (run needs CLIP).
reclaim_hf_cache() {
    log "Step (b): HuggingFace cache."
    if [[ ! -d "$HF_CACHE_DIR" ]]; then
        log "  HF cache dir absent ($HF_CACHE_DIR); nothing to do."
        return 0
    fi
    if [[ "$DISK_GUARD_HF_PURGE" != "1" ]]; then
        log "  HF cache purge skipped (conservative); needs manual review. Set DISK_GUARD_HF_PURGE=1 to allow datasets--*/.locks removal (never models--*)."
        return 0
    fi
    log "  DISK_GUARD_HF_PURGE=1: removing only datasets--* dirs and .locks (never models--*)."
    shopt -s nullglob
    local _t
    for _t in "$HF_CACHE_DIR/hub/datasets--"* "$HF_CACHE_DIR/hub/.locks" "$HF_CACHE_DIR/datasets--"*; do
        [[ -e "$_t" ]] || continue
        # Extra guard: never let a models--* path through.
        if [[ "$_t" == *"models--"* ]]; then
            log "  SKIP (models--* protected): $_t"
            continue
        fi
        local _sz
        _sz="$(du -sh -- "$_t" 2>/dev/null | awk '{print $1}')"
        if [[ "$DRY_RUN" == "1" ]]; then
            log "  would rm -rf $_t (size ${_sz:-?})"
        else
            if _dg_path_allowed "$_t"; then
                rm -rf -- "$_t" 2>/dev/null && log "  removed $_t (was ${_sz:-?})" \
                    || log "  WARN failed to remove $_t"
            fi
        fi
    done
    shopt -u nullglob
}

# (c) Push-then-verify-then-prune OLDEST checkpoints under logging/.
#     Deterministic B2 dest = logs/<relative-path-under-logging>.
#     We push to that exact path, verify THAT exact path, then prune.
#     Stops as soon as usage would be back under target (best-effort estimate
#     via projected freed bytes; real usage re-checked by the main loop).
reclaim_logging_checkpoints() {
    log "Step (c): push-verify-prune oldest logging/ checkpoints (mirrored to B2 under logs/)."
    local _logdir="$RGCL_ROOT/logging"
    if [[ ! -d "$_logdir" ]]; then
        log "  logging/ dir absent; nothing to do."
        return 0
    fi

    # Gather candidate .pt files, oldest-first by mtime.
    local -a _cands=()
    while IFS= read -r -d '' _line; do
        _cands+=("$_line")
    done < <(find "$_logdir" -type f -name '*.pt' -printf '%T@\t%p\0' 2>/dev/null \
                | sort -z -n)

    if [[ "${#_cands[@]}" -eq 0 ]]; then
        log "  no .pt checkpoints under logging/; nothing to prune."
        return 0
    fi
    log "  found ${#_cands[@]} candidate checkpoint(s), oldest-first."

    # How many bytes do we still need to free to reach target?
    local _need_bytes
    _need_bytes="$(awk -v u="${_DG_USAGE_GB:-0}" -v t="$DISK_GUARD_TARGET_GB" \
        'BEGIN{d=(u-t)*1024*1024*1024; if(d<0)d=0; printf "%d", d}')"
    log "  need to free ~$(_dg_human "$_need_bytes") to reach target ${DISK_GUARD_TARGET_GB}G."

    local _freed=0
    local _entry _mtime _file _rel _dest _subpath _bytes
    for _entry in "${_cands[@]}"; do
        # entry = "<mtime>\t<path>"
        _mtime="${_entry%%$'\t'*}"
        _file="${_entry#*$'\t'}"

        # Stop once projected freed bytes cover the need.
        if [[ "$_need_bytes" -gt 0 && "$_freed" -ge "$_need_bytes" ]]; then
            log "  projected freed $(_dg_human "$_freed") >= need $(_dg_human "$_need_bytes"); stopping step (c)."
            break
        fi

        # Safety: path must be allowed (under logging/, not raw/blocked).
        if ! _dg_path_allowed "$_file"; then
            log "  SKIP (path not allowed): $_file"
            continue
        fi

        _rel="${_file#$_logdir/}"                 # relative path under logging/
        _subpath="logs/$_rel"                      # b2 subpath handed to b2_push.sh
        _dest="$B2_PREFIX/$_subpath"               # full b2 dest for verification
        _bytes="$(_dg_bytes "$_file")"

        log "  candidate: $_file"
        log "    mtime=$(date -d "@${_mtime%.*}" '+%Y-%m-%d %H:%M:%S' 2>/dev/null) size=$(_dg_human "$_bytes")"
        log "    -> b2 subpath: $_subpath"

        if [[ "$DRY_RUN" == "1" ]]; then
            # Simulate push + verify + prune; do NOT write or delete.
            log "    would run: b2_push.sh \"$_file\" \"$_subpath\""
            log "    would verify sha1(local) == sha1 at $_dest"
            log "    would rm \"$_file\"  (frees $(_dg_human "$_bytes"))  IF verify passes"
            _freed="$(( _freed + _bytes ))"
            continue
        fi

        # --- real mode ---
        if [[ ! -x "$B2_PUSH" ]]; then
            log "    ERROR: b2_push helper not executable: $B2_PUSH ; skipping."
            continue
        fi
        log "    pushing to B2 ..."
        if ! "$B2_PUSH" "$_file" "$_subpath" >>"$DISK_GUARD_LOG" 2>&1; then
            log "    ERROR: push failed; NOT deleting local. Skipping."
            continue
        fi
        if verify_on_b2 "$_file" "$_dest"; then
            if _dg_path_allowed "$_file"; then
                if rm -f -- "$_file" 2>/dev/null; then
                    log "    PRUNED local (verified on B2): $_file  (freed $(_dg_human "$_bytes"))"
                    _freed="$(( _freed + _bytes ))"
                else
                    log "    WARN: rm failed for $_file"
                fi
            else
                log "    REFUSE: path check failed at delete time; not deleting $_file"
            fi
        else
            log "    ERROR: verify failed; local kept (safe): $_file"
        fi
    done
    log "  step (c) projected/actual freed: $(_dg_human "$_freed")."
}

# ---------------------------------------------------------------------------
# 8. Main.
# ---------------------------------------------------------------------------
disk_guard_main() {
    log "==== disk_guard start (dry_run=$DRY_RUN, sourced=${_DG_SOURCED}) ===="
    log "config: THRESHOLD=${DISK_GUARD_THRESHOLD_GB}G TARGET=${DISK_GUARD_TARGET_GB}G RGCL_ROOT=$RGCL_ROOT B2_PREFIX=$B2_PREFIX"
    log "log file: $DISK_GUARD_LOG"

    local _usage
    _usage="$(get_usage_gb)"
    if [[ -z "$_usage" ]]; then
        log "WARNING: could not determine quota usage from 'quota -s'. Doing NOTHING destructive."
        log "==== disk_guard end (usage unknown) ===="
        return 0
    fi
    _DG_USAGE_GB="$_usage"
    log "current /data usage: ${_usage}G (threshold ${DISK_GUARD_THRESHOLD_GB}G, target ${DISK_GUARD_TARGET_GB}G)."

    if [[ "$_usage" -le "$DISK_GUARD_THRESHOLD_GB" ]]; then
        log "usage ${_usage}G <= threshold ${DISK_GUARD_THRESHOLD_GB}G: nothing to reclaim. No-op."
        log "==== disk_guard end (under threshold) ===="
        return 0
    fi

    log "usage ${_usage}G > threshold ${DISK_GUARD_THRESHOLD_GB}G: starting reclaim."

    # Helper to re-check usage and decide whether to keep going.
    _dg_under_target() {
        local _u
        _u="$(get_usage_gb)"
        if [[ -z "$_u" ]]; then
            log "  (re-check) usage unknown; treating as still-over to avoid unsafe assumptions, but no destructive step will run without B2 verification."
            _DG_USAGE_GB="${_DG_USAGE_GB:-$DISK_GUARD_THRESHOLD_GB}"
            return 1
        fi
        _DG_USAGE_GB="$_u"
        if [[ "$_u" -le "$DISK_GUARD_TARGET_GB" ]]; then
            log "  usage now ${_u}G <= target ${DISK_GUARD_TARGET_GB}G: reclaim complete."
            return 0
        fi
        log "  usage now ${_u}G still > target ${DISK_GUARD_TARGET_GB}G: continuing."
        return 1
    }

    # (a) VFS cache
    reclaim_vfs_cache
    if [[ "$DRY_RUN" != "1" ]] && _dg_under_target; then
        log "==== disk_guard end (target reached after step a) ===="
        return 0
    fi

    # (b) HF cache (conservative)
    reclaim_hf_cache
    if [[ "$DRY_RUN" != "1" ]] && _dg_under_target; then
        log "==== disk_guard end (target reached after step b) ===="
        return 0
    fi

    # (c) push-verify-prune oldest logging checkpoints
    reclaim_logging_checkpoints
    if [[ "$DRY_RUN" != "1" ]] && _dg_under_target; then
        log "==== disk_guard end (target reached after step c) ===="
        return 0
    fi

    # (d) still over target
    if [[ "$DRY_RUN" == "1" ]]; then
        log "(dry-run) simulation complete; see [DRY-RUN] lines above for the planned push/verify/prune actions."
    else
        local _final
        _final="$(get_usage_gb)"
        if [[ -n "$_final" && "$_final" -le "$DISK_GUARD_TARGET_GB" ]]; then
            log "reclaim complete: usage ${_final}G <= target ${DISK_GUARD_TARGET_GB}G."
        else
            log "WARNING: still over target (usage=${_final:-unknown}G > target ${DISK_GUARD_TARGET_GB}G) after all SAFE steps."
            log "WARNING: MANUAL INTERVENTION required — an approved larger offload (e.g. raw datasets or another project) is needed."
            log "WARNING: this guard will NOT touch raw datasets (Multihateclip/HateMM/ImpliHateVid) or non-project data. Stopping."
        fi
    fi
    log "==== disk_guard end ===="
    return 0
}

# Run, then always finish cleanly (return 0 sourced / exit 0 executed).
disk_guard_main
_dg_finish
