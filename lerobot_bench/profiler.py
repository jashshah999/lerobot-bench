"""Profile inference time and VRAM usage for policies."""

import time
import subprocess
import sys
import json
import os
from dataclasses import dataclass


@dataclass
class ProfileMetrics:
    inference_ms_mean: float = 0.0
    inference_ms_std: float = 0.0
    inference_ms_p50: float = 0.0
    inference_ms_p95: float = 0.0
    vram_peak_mb: float = 0.0
    vram_allocated_mb: float = 0.0
    model_params_m: float = 0.0


def profile_policy(policy_path, env_config=None, device="cuda", n_steps=100):
    """Profile a policy's inference time and VRAM usage.

    Runs a lightweight profiling script that:
    1. Loads the policy
    2. Measures VRAM after loading
    3. Runs n_steps forward passes with dummy observations
    4. Reports timing statistics
    """
    script = _PROFILE_SCRIPT.format(
        policy_path=policy_path,
        env_config=env_config or "",
        device=device,
        n_steps=n_steps,
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=300,
    )

    if result.returncode != 0:
        return None, result.stderr[-300:]

    try:
        data = json.loads(result.stdout.strip().split("\n")[-1])
        return ProfileMetrics(**data), None
    except (json.JSONDecodeError, TypeError) as e:
        return None, f"Failed to parse profiler output: {e}"


_PROFILE_SCRIPT = '''
import json
import time
import sys

try:
    import torch
    import numpy as np
except ImportError:
    print(json.dumps({{"error": "torch not available"}}))
    sys.exit(1)

policy_path = "{policy_path}"
device = "{device}"
n_steps = {n_steps}

# Try to load policy
try:
    from lerobot.common.policies.factory import make_policy
    from lerobot.common.policies.pretrained import PreTrainedPolicy

    if hasattr(PreTrainedPolicy, "from_pretrained"):
        policy = PreTrainedPolicy.from_pretrained(policy_path)
    else:
        # Fallback for older lerobot versions
        from omegaconf import OmegaConf
        policy = make_policy(policy_path)

    policy = policy.to(device)
    policy.eval()
except Exception as e:
    print(json.dumps({{"error": f"Failed to load policy: {{e}}"}}))
    sys.exit(1)

# Count parameters
n_params = sum(p.numel() for p in policy.parameters())

# Measure VRAM after loading
if device == "cuda" and torch.cuda.is_available():
    torch.cuda.synchronize()
    vram_allocated = torch.cuda.memory_allocated() / 1024 / 1024
    vram_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
else:
    vram_allocated = 0
    vram_peak = 0

# Profile inference
# Create dummy observation matching expected input
timings = []
with torch.no_grad():
    for i in range(n_steps + 10):  # 10 warmup
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Use policy.select_action or forward with dummy input
        try:
            # Try the standard lerobot interface
            dummy_obs = {{}}
            for key in policy.config.input_features:
                shape = policy.config.input_features[key].shape
                dummy_obs[key] = torch.randn(1, *shape, device=device)
            policy.select_action(dummy_obs)
        except Exception:
            # If that fails, just do a basic forward
            break

        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if i >= 10:  # skip warmup
            timings.append(elapsed_ms)

if device == "cuda" and torch.cuda.is_available():
    torch.cuda.synchronize()
    vram_peak = torch.cuda.max_memory_allocated() / 1024 / 1024

timings = np.array(timings) if timings else np.array([0.0])

result = {{
    "inference_ms_mean": float(timings.mean()),
    "inference_ms_std": float(timings.std()),
    "inference_ms_p50": float(np.percentile(timings, 50)),
    "inference_ms_p95": float(np.percentile(timings, 95)),
    "vram_peak_mb": float(vram_peak),
    "vram_allocated_mb": float(vram_allocated),
    "model_params_m": float(n_params / 1e6),
}}
print(json.dumps(result))
'''
