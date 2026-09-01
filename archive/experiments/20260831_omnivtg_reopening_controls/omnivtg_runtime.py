"""Shared OmniVTG eager-vLLM inference runtime used by approved diagnostics."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from omnivtg_protocol import MODEL_ID, QUERY, parse_interval


TOTAL_VIDEO_TOKENS = 3584
MAX_FRAMES = 768
FPS = 2
PATCH_SIZE = 14
SYSTEM_MESSAGE = "You are a helpful assistant."
PROMPT = """You are given a video as a sequence of interleaved timestamps and frames. Locate the precise timestamps for the event: "{query}".

Use a coarse-to-fine reasoning: first state the broad segment where related content may occur, then refine to localize the secific query. Every reasoning step must include timestamps in the form xx.x seconds.

Wrap the step-by-step reasoning in <think>...</think>. After that, output only the final answer in this exact format inside <answer>...</answer>: From start_time seconds to end_time seconds
For example:
<think>
For the query "a woman opens the door", I find that the woman appears in the video from 12.5 seconds to 20.0 seconds. Zooming in further, she opens the door from 14.2 seconds to 14.7 seconds.
</think>
<answer>
From 14.2 seconds to 14.7 seconds
</answer>"""


def calculate_timestamps(indices, video_fps, merge_size=2):
    indices = indices.tolist() if not isinstance(indices, list) else list(indices)
    if not indices or video_fps <= 0 or merge_size <= 0:
        raise ValueError("timestamps need nonempty indices, positive fps and merge size")
    if len(indices) % merge_size:
        indices.extend(indices[-1] for _ in range(merge_size - len(indices) % merge_size))
    first = indices[0]
    raw = [(index - first) / video_fps for index in indices]
    return [
        (raw[index] + raw[index + merge_size - 1]) / 2
        for index in range(0, len(raw), merge_size)
    ]


def prepare_inputs(video_path: Path, processor):
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [{"role": "user", "content": [{
        "type": "video", "video": str(video_path),
        "min_pixels": 16 * 28 * 28,
        "total_pixels": TOTAL_VIDEO_TOKENS * (PATCH_SIZE * 2) ** 2,
        "max_frames": MAX_FRAMES, "fps": FPS,
    }]}]
    _, videos, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True, return_video_metadata=True
    )
    video_data, metadata = videos[0]
    inputs = processor(
        text="", images=None, videos=[video_data], padding=False,
        do_resize=False, return_tensors="pt", **video_kwargs
    )
    timestamps = calculate_timestamps(metadata["frames_indices"], metadata["fps"], 2)
    frame_count = int(inputs["video_grid_thw"][0, 0])
    if len(timestamps) != frame_count:
        raise RuntimeError("timestamp/frame-grid mismatch")
    im_start, im_end = "<|im_start|>", "<|im_end|>"
    system = f"{im_start}system\n{SYSTEM_MESSAGE}{im_end}\n"
    user = "".join(
        f"<{timestamp:.1f} seconds><|vision_start|><|video_pad|><|vision_end|>"
        for timestamp in timestamps
    )
    user += PROMPT.format(query=QUERY)
    user = f"{im_start}user\n{user}{im_end}\n{im_start}assistant\n"
    input_ids = torch.cat([
        processor.tokenizer(system, add_special_tokens=False, return_tensors="pt")["input_ids"][0],
        processor.tokenizer(user, add_special_tokens=False, return_tensors="pt")["input_ids"][0],
    ])[None]
    video_grid = torch.repeat_interleave(
        inputs["video_grid_thw"], inputs["video_grid_thw"][:, 0], dim=0
    )
    video_grid[:, 0] = 1
    return input_ids[0].tolist(), {
        "video_embeds": None,
        "pixel_values_videos": inputs["pixel_values_videos"],
        "video_grid_thw": video_grid,
        "second_per_grid_ts": [0] * len(video_grid),
    }


def build_model():
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    llm = LLM(
        model=MODEL_ID, tensor_parallel_size=1, trust_remote_code=True,
        enforce_eager=True, max_model_len=32768, max_num_batched_tokens=32768,
        disable_mm_preprocessor_cache=True, gpu_memory_utilization=0.8,
        limit_mm_per_prompt={"image": 0, "video": 768},
        mm_processor_kwargs={"min_pixels": 28 * 28, "max_pixels": 16 * 28 * 28},
    )
    sampling = SamplingParams(
        repetition_penalty=1.05, temperature=0.0, top_p=1.0, top_k=-1,
        stop_token_ids=[151645, 151643], max_tokens=1024,
        include_stop_str_in_output=False, skip_special_tokens=False,
        spaces_between_special_tokens=False,
    )
    return processor, llm, sampling


def infer_one(video_path: Path, processor, llm, sampling):
    prompt_token_ids, mm_data = prepare_inputs(video_path, processor)
    outputs = llm.generate(
        prompts=[{"prompt_token_ids": prompt_token_ids,
                  "multi_modal_data": {"video": mm_data}}],
        sampling_params=sampling, use_tqdm=False,
    )
    completion = outputs[0].outputs[0].text.strip()
    return completion, parse_interval(completion)
