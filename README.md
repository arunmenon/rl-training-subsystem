# RL Training Subsystem – README

Welcome to the **RL Training Subsystem**, an end-to-end workflow for running Reinforcement Learning (RL) jobs—particularly with **GRPO** training—across a distributed fleet of VMs. Below is a comprehensive overview of all major components, including how they fit together, what each does, and how to adapt the system to your needs.

This README is **designed to be very intuitive**, highlighting key architecture points, configuration settings, and extensibility tips.

---

## Table of Contents
1. [High-Level Flow](#high-level-flow)  
2. [Components Overview](#components-overview)  
   1. [Database & Models](#1-database--models)  
   2. [GPU Poller](#2-gpu-poller)  
   3. [Dispatcher](#3-dispatcher)  
   4. [Requeue Manager](#4-requeue-manager)  
   5. [Training Script (`train_grpo.py`)](#5-training-script-train_grpopy)  
   6. [Reward Functions](#6-reward-functions)  
3. [Workflow Summary](#workflow-summary)  
4. [Scaling & Self-Healing](#scaling--self-healing)  
5. [Adding New Reward Functions](#adding-new-reward-functions)  
6. [Usage Examples](#usage-examples)  
7. [Common Customizations](#common-customizations)  
8. [Troubleshooting & Tips](#troubleshooting--tips)

---

## High-Level Flow

Below is a **basic** visual flow of how the system pieces fit together:

```
1. New Job is created (in the DB or published to "jobs.new")

2. The Dispatcher (subscribed to "jobs.new") picks a VM 
   with a free GPU and launches train_grpo.py on that VM.

3. The GPU Poller continually updates each VM's 
   available GPU count in the DB.

4. If a job fails or is pending with no available GPU, 
   the Requeue Manager re-publishes it as "jobs.new" 
   to attempt scheduling again.

5. The training script (train_grpo.py) runs RL 
   training with GRPO (optionally using Unsloth) and 
   saves the final model.
```

---

## Components Overview

### 1) Database & Models
- **`db_models.py`** typically contains SQLAlchemy (or similar) models:
  - **`VM`**: Holds VM connection info (e.g. `ip_address`, `user`, `ssh_key_path`, `available_gpus`).
  - **`Job`**: Represents a training job (`job_id`, `model`, `dataset`, `reward_function`, `status`, `retry_count`, etc.).

These tables are crucial for orchestration: 
- `VM.available_gpus` is updated by the poller, used by dispatcher to schedule jobs.  
- `Job.status` transitions through states like `"queued"` → `"running"` → `"completed"` or `"failed"`.

---

### 2) GPU Poller
- **`gpu_poller.py`**: A process (or service) that periodically **SSH**es into each VM, runs `nvidia-smi` (or a similar command), and parses free GPU counts.  
- Updates `VM.available_gpus` in the database.  
- Uses async or parallel logic so it can handle many VMs without becoming a bottleneck.

---

### 3) Dispatcher
- **`dispatcher.py`** with a class **`GlobalJobDispatcher`**: 
  - Subscribes to **`jobs.new`** (NATS topic) for new job IDs.
  - Picks a VM with free GPUs (via an **atomic** DB update) to prevent race conditions.
  - Sets the job’s status to **`running`** and launches `train_grpo.py` on the VM via SSH.  
  - If no GPU is free, marks job as **`pending`** instead.

---

### 4) Requeue Manager
- A **sidecar or integrated** process that periodically checks:
  - Jobs in **`failed`** status with remaining retries, or  
  - Jobs stuck in **`pending`** for too long.  
- Resets their status to `"queued"`, increments retries if needed, and **re-publishes** them to `"jobs.new"` so the dispatcher can try again.  
- Improves **self-healing** by automatically handling transient failures (e.g. OOM, network hiccups).

---

### 5) Training Script (`train_grpo.py`)
- A **parameterized** script for running RL training:
  - Arguments like `--model`, `--dataset`, `--reward_function`, `--max_steps`, etc.
  - **Preprocessing pipelines** can transform raw data in different ways (e.g., `generic` or `deepseek`).
  - **Reward functions** can be built-in or dynamically imported (`module:function`).
  - Optionally uses **Unsloth** if `--use_unsloth` is passed.
- On completion, saves the trained model + tokenizer to `--output_dir`.

---

### 6) Reward Functions
- Each job references a **reward function** that scores model outputs during GRPO training.
- Built-in examples might be:
  - **`correctness`**: Simple text equality check against reference.  
  - **`keyword_match`**: Looks for certain keywords in output.  
  - **`strict_format`**: Ensures output has a specific XML structure.
- You can define new ones in your own Python modules and load them by passing `--reward_function mymodule:my_reward_func`.

---

## Workflow Summary

1. **Job Created**: Either inserted in the DB or published to **`jobs.new`**.  
2. **Dispatcher** sees the job ID, finds a VM with `available_gpus > 0`, and spawns `train_grpo.py`.  
3. **GPU Poller** is continuously updating VMs so the dispatcher has fresh GPU stats.  
4. **Requeue Manager** re-publishes jobs that are stuck/failed and still eligible for retries.  
5. **Training** finishes, saving model artifacts. The job can then be marked **`completed`** or **`failed`**.

---

## Scaling & Self-Healing

1. **Asynchronous Poller**: Scales to large clusters of VMs.  
2. **Atomic VM Assignment**: Prevents multiple dispatchers from using the same GPU simultaneously.  
3. **Requeue Logic**: Automatically retries jobs that fail or get stuck.  
4. **Adding More VMs**: Just insert a row in `VM` table. The poller starts collecting data, and the dispatcher can schedule jobs there.

---

## Adding New Reward Functions

1. **Implement** a function with signature:  
   ```python
   def my_reward(prompts, completions, **kwargs) -> list[float]:
       ...
   ```  
2. **Reference** it in your job (via the DB or CLI) as `--reward_function mymodule:my_reward`.  
3. If you want it built-in, modify `BUILTIN_REWARDS` in `train_grpo.py` so you can call it by name (`--reward_function my_cool_reward`).

---

## Usage Examples

1. **Submit a New Job**: Insert into `Job` or publish the job ID to NATS:
   ```python
   # Publish job via NATS
   import asyncio
   from nats.aio.client import Client as NATS

   async def publish_job(job_id):
       nc = NATS()
       await nc.connect("nats://localhost:4222")
       await nc.publish("jobs.new", str(job_id).encode())
       await nc.close()

   asyncio.run(publish_job("job123"))
   ```

2. **Run the Poller**:
   ```bash
   python gpu_poller.py
   ```
   This updates `VM.available_gpus` every X seconds.

3. **Dispatcher**:  
   ```bash
   python dispatcher.py
   ```
   Subscribes to `"jobs.new"`, picks a free VM, and launches `train_grpo.py`.

4. **Requeue Manager** is often integrated into `dispatcher.py` or in a separate script that periodically re-queues failed/pending jobs.

5. **Train Script** (manually):
   ```bash
   python train_grpo.py \
     --model facebook/opt-1.3b \
     --dataset /data/my_dataset.json \
     --reward_function correctness \
     --max_steps 300 \
     --use_unsloth
   ```

---

## Common Customizations

- **Multi-GPU**: Modify the dispatcher to pick a specific GPU index from the VM if it has multiple GPUs.  
- **Partial GPU usage**: If you want to schedule multiple jobs per GPU, you need more advanced logic to track GPU memory, not just count.  
- **Advanced Requeue Conditions**: Filter failed jobs by `failure_reason` (e.g., “OOM” vs. “permanent error”).  
- **Logging**: The dispatcher logs job output to `logs/{job_id}.log`. For large scale, consider shipping logs to a central server like Splunk or ELK.

---

## Troubleshooting & Tips

1. **No VM is Picked**: Ensure poller is running and that at least one VM has `available_gpus > 0`.  
2. **Jobs Stuck in `pending`**: Possibly no GPUs are free, or the poller is offline. Requeue manager might re-queue them if it sees them stuck.  
3. **SSH Errors**: Check `ssh -i <key>` usage, correct user, and host. Also verify firewall rules.  
4. **Reward Function Not Found**: Make sure the module is in Python’s path and spelled correctly.  
5. **Unsloth Not Working**: Must install `unsloth` and pass `--use_unsloth`. Otherwise, standard `GRPOTrainer` is used.

---

**We hope this gives you a clear, intuitive grasp of the RL Training Subsystem.** You can easily **extend** it with new reward functions, different scheduling logic, or custom data pipelines—while relying on the same robust, scalable core. Happy training!