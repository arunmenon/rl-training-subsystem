#!/usr/bin/env python
"""
gpu_poller.py

Async GPU poller that connects to multiple VMs in parallel to gather 
GPU usage via nvidia-smi and updates a shared DB record for each VM.
"""

import asyncio
import asyncssh
from datetime import datetime
from db_models import SessionLocal, VM

class GPUPoller:
    def __init__(self, vm_list, poll_interval=60):
        """
        vm_list: list of dicts with VM connection info, e.g. 
                 [{'id': 'vm1', 'host': 'vm1.example.com', 'user': 'username', 'key': '/path/to/key'}, ...]
        poll_interval: polling interval in seconds.
        """
        self.vm_list = vm_list
        self.poll_interval = poll_interval

    async def poll_gpu_status(self, vm):
        """
        Async SSH into a single VM and retrieve GPU status using nvidia-smi.
        Returns (total_gpus, free_gpu_count).
        """
        host = vm['host']
        user = vm.get('user')
        key = vm.get('key')
        password = vm.get('password')

        async with asyncssh.connect(
            host=host, 
            username=user, 
            client_keys=[key] if key else None, 
            password=password
        ) as conn:
            cmd = "nvidia-smi --query-gpu=index,memory.free,memory.total,utilization.gpu --format=csv,noheader"
            result = await conn.run(cmd, check=True)
            output = result.stdout.strip()

        free_gpu_count = 0
        total_gpus = 0
        for line in output.splitlines():
            total_gpus += 1
            parts = [x.strip() for x in line.split(',')]
            if len(parts) >= 4:
                mem_free = int(parts[1].split()[0])
                util = int(parts[3].split()[0])
                if mem_free > 0 and util == 0:
                    free_gpu_count += 1

        return total_gpus, free_gpu_count

    def update_vm_record(self, vm_id, total_gpus, free_gpu_count):
        session = SessionLocal()
        vm_obj = session.query(VM).filter(VM.vm_id == vm_id).first()
        if vm_obj:
            vm_obj.total_gpus = total_gpus
            vm_obj.available_gpus = free_gpu_count
            vm_obj.last_polled = datetime.utcnow()
            session.commit()
        session.close()

    async def poll_and_update(self, vm):
        vm_id = vm['id']
        try:
            total, free = await self.poll_gpu_status(vm)
            self.update_vm_record(vm_id, total, free)
            print(f"[{datetime.utcnow()}] Updated VM {vm_id}: {free}/{total} GPUs free.")
        except Exception as e:
            print(f"Error polling VM {vm_id}: {e}")
            # We could optionally set VM to inactive here if repeated failures

    async def poll_once(self):
        tasks = []
        for vm in self.vm_list:
            tasks.append(self.poll_and_update(vm))
        await asyncio.gather(*tasks)

    async def run_async(self):
        while True:
            await self.poll_once()
            await asyncio.sleep(self.poll_interval)

    def run(self):
        asyncio.run(self.run_async())

if __name__ == "__main__":
    # Example VM list. In production, load from your VM registry/DB
    vm_list_example = [
        {'id': 'vm1', 'host': 'vm1.example.com', 'user': 'user1', 'key': '/path/to/key1'},
        {'id': 'vm2', 'host': 'vm2.example.com', 'user': 'user2', 'key': '/path/to/key2'},
    ]
    poller = GPUPoller(vm_list_example, poll_interval=60)
    poller.run()
