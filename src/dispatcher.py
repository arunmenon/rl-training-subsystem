#!/usr/bin/env python
"""
dispatcher.py

Global job dispatcher that:
  - Subscribes to "jobs.new" messages (NATS).
  - Atomically assigns a free GPU from a VM to the job.
  - Launches the job on that VM via SSH or a remote command.

Also includes a basic job retry logic if desired.
"""

import os
import json
import asyncio
import datetime
import subprocess
from db_models import SessionLocal, Job, VM
from nats.aio.client import Client as NATS

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
        # The actual job params might be in job.config or similar.
        ssh_cmd = (
            f"ssh -i {vm_key} {vm_user}@{vm_host} "
            f"\"CUDA_VISIBLE_DEVICES={gpu_index} python3 /path/to/src/train_grpo.py --model {job.model} "
            f"--dataset {job.dataset} --reward_function {job.reward_function} --max_steps {job.max_steps} "
            f"--learning_rate {job.learning_rate} --batch_size {job.batch_size} "
            f"--preprocessing_pipeline {job.preprocessing_pipeline} --output_dir /tmp/{job.job_id}_output \""
        )
        print(f"[{datetime.datetime.now()}] Launching job {job_id} on VM {vm_record.vm_id} using GPU {gpu_index}")

        with open(f"logs/{job.job_id}.log", "wb") as log_f:
            proc = subprocess.Popen(ssh_cmd, shell=True, stdout=log_f, stderr=log_f)
        print(f"[{datetime.datetime.now()}] Job {job_id} started with PID {proc.pid}")

    def pick_vm_and_decrement_gpu(self):
        """
        Atomically pick a VM with a free GPU, decrement the GPU count, and return that VM record.
        In SQL: 
            UPDATE vms
               SET available_gpus = available_gpus - 1
             WHERE vm_id = (
                 SELECT vm_id
                   FROM vms
                  WHERE available_gpus > 0
                  ORDER BY available_gpus DESC
                  LIMIT 1
             )
         RETURNING vm_id, ...
        """
        session = SessionLocal()
        # The following is pseudo-code for PostgreSQL; adapt to your DB engine
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
        data = msg.data.decode()
        job_id = data.strip()
        print(f"[{datetime.datetime.now()}] New job received: {job_id}")

        vm_record = self.pick_vm_and_decrement_gpu()
        if vm_record is None:
            print(f"[{datetime.datetime.now()}] No available GPU for job {job_id}. Marking as pending.")
            # Optionally set the job status to 'pending' or 'waiting_for_gpu'
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
        nc = NATS()
        await nc.connect(self.nats_url)
        await nc.subscribe("jobs.new", cb=self.handle_new_job)
        print("Global Job Dispatcher is running. Listening on 'jobs.new'...")
        while True:
            await asyncio.sleep(1)

if __name__ == "__main__":
    dispatcher = GlobalJobDispatcher(nats_url="nats://localhost:4222")
    asyncio.run(dispatcher.run())
