import numpy as np
import pandas as pd
import argparse
import os

def process_df(df):
    # Cut - T_start
    return df.iloc[15000-1200:]

def load_example_traces(files, path):
    """
    Load example per-GPU power traces from CSV or .npy files.
    Returns a list of 1D numpy arrays.
    """
    traces = []
    for f in files:
        if f.endswith(".csv"):
            p = os.path.join(path, f)
            df = pd.read_csv(p)
            df = process_df(df)
            # auto-detect power column if possible
            if "pwr" in df.columns:
                traces.append(df["pwr"].values.astype(float))
            else:
                # fallback: first numeric column
                col = df.select_dtypes(include="number").columns[0]
                traces.append(df[col].values.astype(float))
        elif f.endswith(".npy"):
            p = os.path.join(path, f)
            traces.append(np.load(p).astype(float))
        else:
            print(f"Unsupported file type: {f}")
    return traces

def generate_synthetic_training_power_trace(
    step_duration_s=10,
    steps_per_epoch=100,
    forward_duration_ratio = 0.4,
    backward_duration_ratio = 0.4,
    forward_power=360,
    backward_power=380,
    allreduce_power=300,
    noise_frac=0.002
):
    """
    Generate a synthetic power trace for one training epoch.
    
    Parameters:
    - num_gpus: Number of GPUs in the server/rack
    - step_duration_s: Duration of one training step (forward + backward + all-reduce)
    - steps_per_epoch: Total steps in this epoch
    - forward_power: GPU power (W) during forward pass
    - backward_power: GPU power (W) during backward pass
    - allreduce_power: GPU power (W) during gradient sync
    - baseline_server_power: CPU + DRAM power per GPU (W)
    - pue: PUE factor for PDU-level power
    - noise_frac: fraction of Gaussian noise to add
    
    Returns:
    - pandas DataFrame with columns:
      ["timestamp_s", "P_GPU_total_W", "P_IT_W", "P_PDU_W"]
    """
    
    # Define micro-pattern for one step
    # Split step into 3 phases: forward/backward/allreduce
    forward_len = int(step_duration_s * forward_duration_ratio)
    backward_len = int(step_duration_s * backward_duration_ratio)
    allreduce_len = (step_duration_s - forward_len - backward_len)
    
    # Power arrays for one step
    step_pattern = np.concatenate([
        np.full(forward_len, forward_power),
        np.full(backward_len, backward_power),
        np.full(allreduce_len, allreduce_power)
    ])
    
    # Repeat step pattern for all steps
    total_power_gpu = np.tile(step_pattern, steps_per_epoch)
    
    # Optionally add small Gaussian noise
    noise = np.random.normal(0, noise_frac * total_power_gpu)
    total_power_gpu_noisy = total_power_gpu + noise
    
    # Clip negative values
    total_power_gpu_noisy = np.clip(total_power_gpu_noisy, 0, None)

    return total_power_gpu_noisy

def random_time_shift(trace, max_shift):
    """
    Randomly shifts a trace in time (prepend zeros).
    max_shift: number of seconds to shift at most.
    """
    shift = np.random.randint(0, max_shift + 1)
    return np.concatenate([np.zeros(shift), trace])


def resample_to_length(arr, start_idx, length):
    """Pad or truncate array to desired length."""
    if len(arr) >= length:
        return arr[start_idx:start_idx+length]
    else:
        print("T > T_data: Appending zeros to the end of the array and setting start_idx = 0...")
        return np.concatenate([arr, np.zeros(length - len(arr))])

def resample(arr, old_dt, new_dt):
    """Resample the array at the new sample rate. Linearly interpolates new data points"""
    T_arr = len(arr)
    ts_old = np.arange(0, T_arr*old_dt, old_dt)
    ts_new = np.arange(0, T_arr*old_dt, new_dt)
    new_arr = np.interp(ts_new, ts_old, arr)
    return new_arr

def generate_training_pdu_trace(
    gpu_traces,
    num_gpus=8,
    duration_s=36000,
    start_idx=0,
    server_base_power=50,   # baseline server watts (CPU, DRAM, NIC)
    noise_frac=0.01,         # 1% Gaussian noise
    pue=1.2,                 # apply PUE to create PDU-level power
    max_shift=10,
    gpu_trace_map = None
):
    """
    Create a synthetic PDU-level power trace for a rack of GPUs.
    """

    # Prepare output container
    t = np.arange(duration_s)
    gpu_power_matrix = np.zeros((num_gpus, duration_s))

    for g in range(num_gpus):
        # pick an example per-GPU trace
        if gpu_trace_map is None:
            trace_idx = np.random.choice(range(len(gpu_traces)))
        else:
            trace_idx = gpu_trace_map[g]
        ex = gpu_traces[trace_idx]
        # random time shift to avoid perfect alignment
        ex_shift = random_time_shift(ex, max_shift=max_shift)

        # resample to desired duration
        ex_resized = resample_to_length(ex_shift, start_idx, duration_s)

        gpu_power_matrix[g] = ex_resized

    # Sum GPU power
    P_GPU_total = gpu_power_matrix.sum(axis=0)

    # Add server baseline
    P_IT = P_GPU_total + num_gpus * server_base_power

    # Add Gaussian noise
    noise = np.random.normal(0, noise_frac * P_IT)
    P_IT_noisy = np.maximum(0, P_IT + noise)

    # Convert to PDU-level power via PUE
    P_PDU = P_IT_noisy * pue

    df = pd.DataFrame({
        "timestamp_s": t,
        "P_GPU_total_W": P_GPU_total,
        "P_IT_W": P_IT_noisy,
        "P_PDU_W": P_PDU
    })

    return df


def inference_power_model(tokens, base_idle=45, peak=280):
    """
    Return a per-second power profile for an inference request.
    - tokens: number of output tokens
    - base_idle: idle GPU power (W)
    - peak: peak inference power (W)

    Inference power typically rises during token generation.
    """
    duration = tokens  # assume 1 token/sec for demonstration
    t = np.arange(duration)
    # Simple model: power ramps up, stays high, then falls
    curve = base_idle + (peak - base_idle) * np.exp(-0.5 * (t - duration/3)**2 / (duration/8)**2)
    return curve


def generate_request_arrivals(duration_s, rate_rps):
    """
    Poisson arrival model.
    Returns a sorted list of arrival timestamps (integers).
    """
    num_expected = int(duration_s * rate_rps * 1.5)
    arrivals = np.cumsum(np.random.exponential(scale=1/rate_rps, size=num_expected))
    arrivals = arrivals[arrivals < duration_s]
    return arrivals.astype(int)

def generate_inference_pdu_trace(
        duration_s=3600,
        num_gpus=8,
        arrival_rate=0.2,  # 0.2 requests/sec = 12 RPM
        server_base_power=120,
        pue=1.2):

    # 1-second resolution
    t = np.arange(duration_s)
    gpu_power = np.zeros((num_gpus, duration_s))

    # A) Generate request arrivals
    arrivals = generate_request_arrivals(duration_s, arrival_rate)

    # B) Choose random per-request token counts
    # Real data: tokens ~ lognormal or exponential distribution
    token_counts = np.random.lognormal(mean=3.5, sigma=0.6, size=len(arrivals)).astype(int)

    # C) Assign requests to GPUs (round-robin or least-loaded)
    next_gpu = 0

    for arrival, tokens in zip(arrivals, token_counts):
        gpu_id = next_gpu
        next_gpu = (next_gpu + 1) % num_gpus

        # Compute power profile for this request
        trace = inference_power_model(tokens)

        # Overlay onto the GPU timeline
        end = min(arrival + len(trace), duration_s)
        gpu_power[gpu_id, arrival:end] += trace[:end - arrival]

    # Aggregate GPU power
    P_GPU_total = gpu_power.sum(axis=0)

    # Add server baseline
    P_IT = P_GPU_total + num_gpus * server_base_power

    # PDU-level
    P_PDU = P_IT * pue

    return pd.DataFrame({
        "timestamp_s": t,
        "P_GPU_total_W": P_GPU_total,
        "P_IT_W": P_IT,
        "P_PDU_W": P_PDU
    })

