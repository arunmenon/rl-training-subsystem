#!/usr/bin/env python
"""
dispatcher.py

Global job dispatcher that:
  - Subscribes to "jobs.new" messages (NATS).
  - Atomically assigns a free GPU from a VM to the job.
  - Launches the job on that VM via SSH or a remote command.

Includes a basic JobRequeueManager that periodically looks for failed/pending 
jobs and re-publishes them if they meet certain retry conditions.
"""

import os
import json
import asyncio
import datetime
import subprocess

from nats.aio.client import Client as NATS

from db_models import SessionLocal, Job, VM

# -----------------------------------------
# Modular Requeue Logic
# -----------------------------------------
class JobRequeueManager:
    """
    Periodically scans for jobs in 'failed' or 'pending' status, 
    and re-publishes them on the "jobs.new" queue if they haven't 
    exceeded a retry limit or if certain conditions are met.
    """

    def __init__(self, nats_url="nats://localhost:4222", poll_interval=60, max_retries=2):
        self.nats_url = nats_url
        self.poll_interval = poll_interval
        self.max_retries = max_retries

    async def run(self):
        """
        Main loop: connect to NATS, then periodically check the DB for 
        any jobs to requeue. If found, publish them to 'jobs.new'.
        """
        self.nc = NATS()
        await self.nc.connect(self.nats_url)

        print(f"[{datetime.datetime.now()}] JobRequeueManager started. Checking for failed/pending jobs every {self.poll_interval}s.")
        try:
            while True:
                await self.check_and_requeue_jobs()
                await asyncio.sleep(self.poll_interval)
        finally:
            await self.nc.close()

    async def check_and_requeue_jobs(self):
        """
        Finds jobs that are 'failed' or 'pending' and re-queues them 
        if they haven't exceeded max retries or meet other criteria.
        """
        session = SessionLocal()

        # Example logic:
        #  1) 'failed' jobs with retry_count < max_retries 
        #  2) 'pending' jobs that are stuck for a long time, or also just re-queue them
        failed_jobs = session.query(Job).filter(
            Job.status == "failed",
            Job.retry_count < self.max_retries
        ).all()

        # Optionally handle 'pending' jobs that might be stuck
        # This is up to your workflow. For example:
        pending_jobs = session.query(Job).filter(
            Job.status == "pending"
        ).all()

        for job in failed_jobs + pending_jobs:
            # Increment retry count if it's a failed job
            if job.status == "failed":
                job.retry_count += 1

            # Move status to 'queued' so dispatcher picks it up
            job.status = "queued"
            job.failure_reason = None  # clearing old reason
            session.commit()

            # Now re-publish job ID to NATS
            print(f"[{datetime.datetime.now()}] Re-queuing job {job.job_id} (attempt {job.retry_count}).")
            await self.nc.publish("jobs.new", str(job.job_id).encode("utf-8"))

        session.close()


# -----------------------------------------
# Main Dispatcher Logic
# -----------------------------------------
class GlobalJobDispatcher:
    def __init__(self, nats_url="nats://localhost:4222"):
        self.nats_url = nats_url

    async def launch_job_on_vm(self, job_id, vm_record):
        """
        Launch a training job on the assigned VM, specifying the GPU index 
        (here we assume GPU 0 if multiple are free, but you can refine).
        """
        vm_host = vm_record.ip_address
        vm_user = vm_record.user
        vm_key = vm_record.ssh_key_path
        gpu_index = 0  # Simplified approach

        session = SessionLocal()
        job = session.query(Job).filter(Job.job_id == job_id).first()
        if job:
            job.assigned_vm = vm_record.vm_id
            job.gpu_index = gpu_index
            job.status = "running"
            job.start_time = datetime.datetime.utcnow()
            job.log_path = f"logs/{job.job_id}.log"
            session.commit()
        session.close()

        # Example remote command: run train_grpo.py with the job's parameters
        ssh_cmd = (
            f"ssh -i {vm_key} {vm_user}@{vm_host} "
            f"\"CUDA_VISIBLE_DEVICES={gpu_index} python3 /path/to/src/train_grpo.py "
            f"--model {job.model} "
            f"--dataset {job.dataset} "
            f"--reward_function {job.reward_function} "
            f"--max_steps {job.max_steps} "
            f"--learning_rate {job.learning_rate} "
            f"--batch_size {job.batch_size} "
            f"--preprocessing_pipeline {job.preprocessing_pipeline} "
            f"--output_dir /tmp/{job.job_id}_output \""
        )
        print(f"[{datetime.datetime.now()}] Launching job {job_id} on VM {vm_record.vm_id} using GPU {gpu_index}")

        with open(f"logs/{job.job_id}.log", "wb") as log_f:
            proc = subprocess.Popen(ssh_cmd, shell=True, stdout=log_f, stderr=log_f)
        print(f"[{datetime.datetime.now()}] Job {job_id} started with PID {proc.pid}")

    def pick_vm_and_decrement_gpu(self):
        """
        Atomically pick a VM with a free GPU, decrement the GPU count, and return that VM record.
        In SQL (PostgreSQL example):
            WITH picked_vm AS (
                UPDATE vms
                SET available_gpus = available_gpus - 1
                WHERE vm_id = (
                    SELECT vm_id 
                    FROM vms
                    WHERE available_gpus > 0
                    ORDER BY available_gpus DESC
                    LIMIT 1
                )
                RETURNING *
            )
            SELECT * FROM picked_vm;
        """
        session = SessionLocal()
        result = session.execute("""
            WITH picked_vm AS (
                UPDATE vms
                SET available_gpus = available_gpus - 1
                WHERE vm_id = (
                    SELECT vm_id 
                    FROM vms
                    WHERE available_gpus > 0
                    ORDER BY available_gpus DESC
                    LIMIT 1
                )
                RETURNING *
            )
            SELECT * FROM picked_vm;
        """)

        row = result.fetchone()
        if row:
            vm_record = VM(
                vm_id=row.vm_id,
                ip_address=row.ip_address,
                user=row.user,
                ssh_key_path=row.ssh_key_path,
                available_gpus=row.available_gpus,
                total_gpus=row.total_gpus,
                last_polled=row.last_polled
            )
            session.commit()
            session.close()
            return vm_record
        else:
            session.close()
            return None

    async def handle_new_job(self, msg):
        """
        Callback when we receive a 'jobs.new' message. 
        Attempts to pick a VM and start the job. Otherwise, sets it to 'pending'.
        """
        data = msg.data.decode()
        job_id = data.strip()
        print(f"[{datetime.datetime.now()}] New job received: {job_id}")

        vm_record = self.pick_vm_and_decrement_gpu()
        if vm_record is None:
            print(f"[{datetime.datetime.now()}] No available GPU for job {job_id}. Marking as pending.")
            session = SessionLocal()
            job = session.query(Job).filter(Job.job_id == job_id).first()
            if job:
                job.status = "pending"
                session.commit()
            session.close()
            return

        # Otherwise, launch
        asyncio.create_task(self.launch_job_on_vm(job_id, vm_record))

    async def run(self):
        """
        Connect to NATS, subscribe to 'jobs.new', and handle new jobs in an infinite loop.
        """
        nc = NATS()
        await nc.connect(self.nats_url)
        await nc.subscribe("jobs.new", cb=self.handle_new_job)
        print(f"[{datetime.datetime.now()}] Global Job Dispatcher is running. Listening on 'jobs.new'...")
        try:
            while True:
                await asyncio.sleep(1)
        finally:
            await nc.close()

# -----------------------------------------
# Main entry point
# -----------------------------------------
async def main():
    dispatcher = GlobalJobDispatcher(nats_url="nats://localhost:4222")
    requeue_manager = JobRequeueManager(nats_url="nats://localhost:4222", poll_interval=60, max_retries=2)

    # Run both dispatcher and requeue manager concurrently
    await asyncio.gather(
        dispatcher.run(),
        requeue_manager.run(),
    )

if __name__ == "__main__":
    asyncio.run(main())
