# src/dispatcher.py
import os
import json
import asyncio
import datetime
import subprocess
from db_models import SessionLocal, Job, VM
from nats.aio.client import Client as NATS

# Placeholder functions for DB queries/updates.
def db_query_one(query, params):
    session = SessionLocal()
    vm = session.query(VM).filter(VM.available_gpus > 0).first()
    session.close()
    return vm

def db_update(query, params):
    session = SessionLocal()
    # Implement your update logic using SQLAlchemy here.
    session.commit()
    session.close()

class GlobalJobDispatcher:
    def __init__(self, nats_url="nats://localhost:4222"):
        self.nats_url = nats_url

    async def launch_job_on_vm(self, job_id, vm_record):
        """Launch a training job on the assigned VM via SSH."""
        # In a real system, use vm_record's connection details.
        vm_host = vm_record.ip_address
        vm_user = "your_vm_user"         # Replace with actual user or retrieve from vm_record.
        vm_key = "/path/to/your/ssh_key"   # Replace or retrieve from vm_record.
        gpu_index = 0  # For simplicity, we assign GPU index 0.

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

        ssh_cmd = f"CUDA_VISIBLE_DEVICES={gpu_index} python3 /path/to/src/train_grpo.py --job_id {job_id}"
        print(f"[{datetime.datetime.now()}] Launching job {job_id} on VM {vm_record.vm_id} using GPU {gpu_index}")
        proc = subprocess.Popen(ssh_cmd, shell=True)
        print(f"[{datetime.datetime.now()}] Job {job_id} started with PID {proc.pid}")

    async def handle_new_job(self, msg):
        data = msg.data.decode()
        print(f"[{datetime.datetime.now()}] New job received: {data}")
        job_id = data.strip()  # Assuming the message contains the job_id.
        vm = db_query_one("SELECT * FROM vms WHERE available_gpus > 0 ORDER BY available_gpus DESC LIMIT 1", None)
        if vm is None:
            print(f"[{datetime.datetime.now()}] No available GPU for job {job_id}.")
            return
        db_update("UPDATE vms SET available_gpus = available_gpus - 1 WHERE vm_id = %s", (vm.vm_id,))
        asyncio.create_task(self.launch_job_on_vm(job_id, vm))

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
