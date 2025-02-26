
```markdown
# RL-Training-Subsystem

The **RL-Training-Subsystem** is a distributed framework for managing reinforcement learning (GRPO) fine-tuning jobs across multiple GPU-equipped VMs. It orchestrates job scheduling, GPU resource monitoring, and job tracking using a combination of centralized and decentralized components.

## Overview

This system comprises the following key components:

1. **Database Models (`src/db_models.py`):**  
   Defines the SQLAlchemy models for two tables:
   - **`jobs` Table:** Stores job metadata including job ID, model, dataset path, hyperparameters, reward function, status, and assignment details.
   - **`vms` Table:** Records VM information such as unique VM ID, IP address, total GPUs, available GPUs, and the last time the VM was polled.

2. **Centralized GPU Poller (`src/gpu_poller.py`):**  
   A service that runs on a central management host. It periodically SSHes into each registered VM, runs `nvidia-smi` to determine GPU availability, and updates the `vms` table in the database. This keeps a real-time view of available resources.

3. **Global Job Dispatcher (`src/dispatcher.py`):**  
   A centralized component that listens for new job submission messages on the NATS subject `jobs.new`. When a new job is received, it queries the database for a VM with free GPUs, reserves a GPU, and remotely launches the training job (via SSH) on that VM.

4. **Parameterized GRPO Training Script (`src/train_grpo.py`):**  
   This script is executed on the target VM to fine-tune a model using GRPO. It is fully parameterized – accepting the model, dataset, reward function, preprocessing pipeline, and hyperparameters as command-line arguments. Reward functions and preprocessing pipelines can be dynamically loaded.

5. **Job Tracking Daemon (`src/job_tracker.py`):**  
   A daemon that runs on each VM to monitor active training jobs. It periodically checks if the jobs (tracked by their process IDs) are still running. When a job completes or fails, it publishes a message to NATS (on `jobs.completed` or `jobs.errored`) with job metadata, including exit codes and error snippets. The daemon uses a local JSON file for persistent tracking and is designed for self-healing (recommended to run as a systemd service).

6. **Systemd Service (`systemd/job_tracker.service`):**  
   A sample systemd unit file to run the Job Tracking Daemon automatically on each VM. This ensures that job monitoring is always active and auto-restarts if it crashes.

## Database Schema

### Jobs Table

| Column Name       | Data Type | Description                                                                                                     |
|-------------------|-----------|-----------------------------------------------------------------------------------------------------------------|
| **id**            | INTEGER   | Primary key (auto-incremented).                                                                                 |
| **job_id**        | STRING    | Unique job identifier (default: UUID).                                                                          |
| **job_type**      | STRING    | Type of job (e.g., "SFT" for supervised fine-tuning, "GRPO" for RL fine-tuning).                                |
| **model**         | STRING    | Name or path of the base model used for training.                                                               |
| **dataset_path**  | STRING    | Path or identifier for the dataset.                                                                             |
| **hyperparams**   | JSON      | Hyperparameters (e.g., learning rate, batch size, max steps) stored as a JSON object.                             |
| **reward_function** | STRING  | Name or specification for the reward function (e.g., "correctness", "keyword_match", or "module:function").        |
| **output_dir**    | STRING    | Directory where job outputs and checkpoints will be saved.                                                      |
| **status**        | STRING    | Current job status: "queued", "running", "completed", or "failed".                                               |
| **assigned_vm**   | STRING    | Identifier of the VM where the job is assigned (nullable until assignment).                                     |
| **gpu_index**     | INTEGER   | GPU index on the assigned VM (nullable until assignment).                                                       |
| **created_at**    | DATETIME  | Timestamp when the job was created.                                                                             |
| **start_time**    | DATETIME  | Timestamp when the job started running (nullable until the job starts).                                          |
| **end_time**      | DATETIME  | Timestamp when the job completed or failed (nullable until the job ends).                                         |
| **log_path**      | STRING    | File path to the job log (nullable until set).                                                                  |
| **priority**      | INTEGER   | Optional priority level for scheduling (default is 0).                                                          |

### VMs Table

| Column Name        | Data Type | Description                                                                                                     |
|--------------------|-----------|-----------------------------------------------------------------------------------------------------------------|
| **id**             | INTEGER   | Primary key (auto-incremented).                                                                                 |
| **vm_id**          | STRING    | Unique identifier for the VM (e.g., hostname).                                                                  |
| **ip_address**     | STRING    | IP address of the VM.                                                                                           |
| **total_gpus**     | INTEGER   | Total number of GPUs available on the VM.                                                                       |
| **available_gpus** | INTEGER   | Number of GPUs currently free.                                                                                  |
| **active**         | BOOLEAN   | Indicates if the VM is active/online (default: true).                                                           |
| **last_polled**    | DATETIME  | Timestamp of the last time the VM's GPU status was updated.                                                     |

## Flow Description

1. **Job Submission:**  
   - A client or scheduler inserts a job record into the `jobs` table and publishes a message with the job ID to the NATS subject `jobs.new`.

2. **Resource Polling:**  
   - The Centralized GPU Poller periodically (every 60 seconds by default) SSHes into each VM, runs `nvidia-smi`, and updates the `vms` table with the number of free GPUs.

3. **Job Dispatching:**  
   - The Global Job Dispatcher listens on `jobs.new`. When a new job message arrives, it queries the `vms` table to find a VM with available GPUs.
   - Once a VM is selected, the dispatcher reserves a GPU (decrements the available GPU count in the DB) and launches the training job remotely on that VM using SSH.
   - The training job runs the parameterized `train_grpo.py` script with the required parameters.

4. **Job Tracking:**  
   - The Job Tracking Daemon on each VM monitors the processes for active jobs by checking their PIDs. It runs every minute (configurable).
   - When a job finishes successfully, the daemon publishes a message to NATS on `jobs.completed`. If a job fails, it publishes to `jobs.errored` with details such as exit code and error snippets.
   - The job tracking state is persisted locally so that if the daemon restarts, it can recover from the last state.

5. **Monitoring & Logging:**  
   - All components write logs (e.g., to `/var/log/`). The system is designed to be robust and self-healing. The Job Tracker is managed by systemd to auto-restart in case of failure.

## Setup & Deployment

1. **Database:**  
   - Update `DATABASE_URL` in `src/db_models.py` with your PostgreSQL connection string.
   - Run the following command to create the necessary tables:
     ```bash
     python src/db_models.py
     ```

2. **Dependencies:**  
   - Install Python dependencies using:
     ```bash
     pip install -r requirements.txt
     ```

3. **Centralized GPU Poller:**  
   - Configure the VM list in `src/gpu_poller.py` with your VM connection details.
   - Run the poller on a central host:
     ```bash
     python src/gpu_poller.py
     ```

4. **Global Job Dispatcher:**  
   - Start the dispatcher on a central management node:
     ```bash
     python src/dispatcher.py
     ```

5. **Job Tracking Daemon:**  
   - On each VM that will run training jobs, install and start the Job Tracker.
   - If using systemd, place the `systemd/job_tracker.service` file in `/etc/systemd/system/`, then run:
     ```bash
     sudo systemctl enable job_tracker
     sudo systemctl start job_tracker
     ```

6. **Training Script:**  
   - The dispatcher launches the training script (`src/train_grpo.py`) on the target VM. You can also run it manually for testing:
     ```bash
     python src/train_grpo.py --model facebook/opt-1.3b --dataset /path/to/data.json --reward_function correctness --preprocessing_pipeline title --max_steps 300 --learning_rate 1e-5 --batch_size 2
     ```

7. **NATS Server:**  
   - Ensure you have a running NATS server (default URL: `nats://localhost:4222`). Update the NATS URL in the code if necessary.

## Detailed Flow

- **Job Submission:**  
  A job is submitted by creating a record in the `jobs` table (with details such as the model to use, dataset path, hyperparameters, etc.) and then publishing its `job_id` to the NATS subject `jobs.new`. This signals the system that a new training job is waiting to be processed.

- **Resource Polling:**  
  The Centralized GPU Poller runs on a central server. It uses SSH to log into each VM and executes `nvidia-smi` to check how many GPUs are free. This information is then written to the `vms` table in the database. This way, the system always knows which VMs have available resources.

- **Job Dispatching:**  
  The Global Job Dispatcher listens for new job notifications. When it receives a job ID, it looks up the `vms` table to select a VM with free GPUs. It then reserves a GPU by decrementing the available GPU count in the database and launches the training job remotely on that VM via SSH. The job executes the `train_grpo.py` script with all necessary parameters.

- **Job Tracking:**  
  Each VM runs a Job Tracking Daemon, which monitors all training jobs on that VM. It periodically checks the process IDs of active jobs. When it detects that a job has finished (or crashed), it publishes a corresponding message (either to `jobs.completed` for successful completions or `jobs.errored` for failures) to NATS. This decentralized tracking ensures that the central system is promptly updated about job outcomes without burdening the global dispatcher.

- **Monitoring & Logging:**  
  All services log their activities to local log files (e.g., in `/var/log/`). The Job Tracking Daemon is set up to auto-restart via systemd if it crashes, ensuring continuous monitoring.

## Extensibility & Error Handling

- **Parameterization:**  
  The training script (`train_grpo.py`) is fully parameterized. You can change reward functions, preprocessing pipelines, and hyperparameters using command-line arguments. This allows for easy experimentation without modifying the code.

- **Dynamic Imports:**  
  Reward functions and preprocessing pipelines can be extended by adding new modules. The system dynamically loads these based on the parameters provided at runtime.

- **Robust Messaging:**  
  The use of NATS for messaging decouples the components, so each part of the system (submission, dispatch, tracking) can operate independently and reliably.


```

