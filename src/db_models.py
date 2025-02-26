# src/db_models.py
import datetime
import uuid
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Update DATABASE_URL with your PostgreSQL connection string.
DATABASE_URL = "postgresql://user:pass@localhost:5432/jobsdb"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Job(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    job_type = Column(String)  # e.g., 'SFT' or 'GRPO'
    model = Column(String)
    dataset_path = Column(String)
    hyperparams = Column(JSON)  # Hyperparameters stored as JSON.
    reward_function = Column(String)  # Name or spec for the reward function.
    output_dir = Column(String)
    status = Column(String, default="queued")  # queued, running, completed, failed.
    assigned_vm = Column(String, nullable=True)
    gpu_index = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    log_path = Column(String, nullable=True)
    priority = Column(Integer, default=0)

class VM(Base):
    __tablename__ = "vms"
    id = Column(Integer, primary_key=True, index=True)
    vm_id = Column(String, unique=True, index=True)  # Unique identifier (e.g., hostname)
    ip_address = Column(String)
    total_gpus = Column(Integer, default=0)
    available_gpus = Column(Integer, default=0)  # Number of free GPUs
    active = Column(Boolean, default=True)
    last_polled = Column(DateTime, default=datetime.datetime.utcnow)

def create_tables():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    create_tables()
    print("Database tables created.")
