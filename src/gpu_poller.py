# src/gpu_poller.py
import paramiko
import time
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

    def poll_gpu_status(self, vm):
        """SSH into a single VM and retrieve GPU status using nvidia-smi."""
        host = vm['host']
        user = vm.get('user')
        key = vm.get('key')
        password = vm.get('password')

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=host, username=user, key_filename=key, password=password)
        # Query GPU info in CSV format (index, free memory, total memory, utilization)
        cmd = "nvidia-smi --query-gpu=index,memory.free,memory.total,utilization.gpu --format=csv,noheader"
        stdin, stdout, stderr = ssh.exec_command(cmd)
        output = stdout.read().decode().strip()
        ssh.close()

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
        now = datetime.utcnow()
        session = SessionLocal()
        vm = session.query(VM).filter(VM.vm_id == vm_id).first()
        if vm:
            vm.total_gpus = total_gpus
            vm.available_gpus = free_gpu_count
            vm.last_polled = now
            session.commit()
        session.close()

    def run(self):
        while True:
            for vm in self.vm_list:
                vm_id = vm['id']
                try:
                    total, free = self.poll_gpu_status(vm)
                    self.update_vm_record(vm_id, total, free)
                    print(f"[{datetime.utcnow()}] Updated VM {vm_id}: {free}/{total} GPUs free.")
                except Exception as e:
                    print(f"Error polling VM {vm_id}: {e}")
            time.sleep(self.poll_interval)

if __name__ == "__main__":
    # Example VM list. In production, load from your VM registry.
    vm_list = [
        {'id': 'vm1', 'host': 'vm1.example.com', 'user': 'user1', 'key': '/path/to/key1'},
        {'id': 'vm2', 'host': 'vm2.example.com', 'user': 'user2', 'key': '/path/to/key2'},
    ]
    poller = GPUPoller(vm_list, poll_interval=60)
    poller.run()
