"""
Job Database: SQLite-based tracking for Vast.ai jobs.

Tracks all jobs (test, eval, finetune) with their instance IDs,
status, and results - so you never lose track of running instances.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class JobStatus(Enum):
    PENDING = "pending"
    CREATING = "creating"
    UPLOADING = "uploading"
    RUNNING = "running"
    DOWNLOADING = "downloading"
    SCORING = "scoring"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(Enum):
    TEST = "test"
    EVAL = "eval"
    FINETUNE = "finetune"
    BATCH = "batch"


@dataclass
class Job:
    """Represents a Vast.ai job."""
    job_id: str
    job_type: str
    model_id: str
    status: str
    instance_id: Optional[int] = None
    ssh_host: Optional[str] = None
    ssh_port: Optional[int] = None
    preset: Optional[str] = None
    samples: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    completed_at: Optional[str] = None
    cost_estimate: Optional[float] = None
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Optional[str] = None  # JSON string for extra data

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Job":
        return cls(**dict(row))


class JobDatabase:
    """SQLite database for tracking Vast.ai jobs."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            # Default to outputs/vastai/jobs.db
            db_path = Path(__file__).parent.parent.parent / "outputs" / "vastai" / "jobs.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    instance_id INTEGER,
                    ssh_host TEXT,
                    ssh_port INTEGER,
                    preset TEXT,
                    samples INTEGER,
                    created_at TEXT,
                    updated_at TEXT,
                    completed_at TEXT,
                    cost_estimate REAL,
                    output_path TEXT,
                    error_message TEXT,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON jobs(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_instance ON jobs(instance_id)
            """)
            conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def generate_job_id(self, job_type: str) -> str:
        """Generate a unique job ID."""
        import hashlib
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique = hashlib.md5(f"{timestamp}{job_type}".encode()).hexdigest()[:6]
        return f"{job_type}-{unique}"

    def create_job(
        self,
        job_type: JobType,
        model_id: str,
        preset: Optional[str] = None,
        samples: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> Job:
        """Create a new job record."""
        job_id = self.generate_job_id(job_type.value)
        now = datetime.now().isoformat()

        job = Job(
            job_id=job_id,
            job_type=job_type.value,
            model_id=model_id,
            status=JobStatus.PENDING.value,
            preset=preset,
            samples=samples,
            created_at=now,
            updated_at=now,
            metadata=json.dumps(metadata) if metadata else None
        )

        with self._get_conn() as conn:
            conn.execute("""
                INSERT INTO jobs (
                    job_id, job_type, model_id, status, preset, samples,
                    created_at, updated_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job.job_id, job.job_type, job.model_id, job.status,
                job.preset, job.samples, job.created_at, job.updated_at,
                job.metadata
            ))
            conn.commit()

        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            return Job.from_row(row) if row else None

    def get_job_by_instance(self, instance_id: int) -> Optional[Job]:
        """Get a job by instance ID."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE instance_id = ?", (instance_id,)
            ).fetchone()
            return Job.from_row(row) if row else None

    def update_job(self, job_id: str, **kwargs) -> bool:
        """Update job fields."""
        kwargs["updated_at"] = datetime.now().isoformat()

        # Handle metadata specially
        if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
            kwargs["metadata"] = json.dumps(kwargs["metadata"])

        fields = ", ".join(f"{k} = ?" for k in kwargs.keys())
        values = list(kwargs.values()) + [job_id]

        with self._get_conn() as conn:
            cursor = conn.execute(
                f"UPDATE jobs SET {fields} WHERE job_id = ?", values
            )
            conn.commit()
            return cursor.rowcount > 0

    def set_instance(self, job_id: str, instance_id: int, ssh_host: str, ssh_port: int):
        """Set instance connection details."""
        self.update_job(
            job_id,
            instance_id=instance_id,
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            status=JobStatus.CREATING.value
        )

    def set_status(self, job_id: str, status: JobStatus, error_message: Optional[str] = None):
        """Update job status."""
        updates = {"status": status.value}
        if status in (JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.CANCELLED):
            updates["completed_at"] = datetime.now().isoformat()
        if error_message:
            updates["error_message"] = error_message
        self.update_job(job_id, **updates)

    def set_output(self, job_id: str, output_path: str):
        """Set job output path."""
        self.update_job(job_id, output_path=output_path)

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        job_type: Optional[JobType] = None,
        limit: int = 50
    ) -> List[Job]:
        """List jobs with optional filtering."""
        query = "SELECT * FROM jobs"
        params = []
        conditions = []

        if status:
            conditions.append("status = ?")
            params.append(status.value)
        if job_type:
            conditions.append("job_type = ?")
            params.append(job_type.value)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [Job.from_row(row) for row in rows]

    def get_active_jobs(self) -> List[Job]:
        """Get all jobs that are currently active (not complete/failed/cancelled)."""
        active_statuses = [
            JobStatus.PENDING.value,
            JobStatus.CREATING.value,
            JobStatus.UPLOADING.value,
            JobStatus.RUNNING.value,
            JobStatus.DOWNLOADING.value,
            JobStatus.SCORING.value,
        ]
        placeholders = ",".join("?" * len(active_statuses))

        with self._get_conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM jobs WHERE status IN ({placeholders}) ORDER BY created_at DESC",
                active_statuses
            ).fetchall()
            return [Job.from_row(row) for row in rows]

    def get_running_instances(self) -> List[int]:
        """Get all instance IDs that are currently running."""
        jobs = self.get_active_jobs()
        return [j.instance_id for j in jobs if j.instance_id]

    def delete_job(self, job_id: str) -> bool:
        """Delete a job record."""
        with self._get_conn() as conn:
            cursor = conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            conn.commit()
            return cursor.rowcount > 0

    def cleanup_old_jobs(self, days: int = 30) -> int:
        """Delete jobs older than N days."""
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM jobs WHERE created_at < ? AND status IN (?, ?, ?)",
                (cutoff, JobStatus.COMPLETE.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value)
            )
            conn.commit()
            return cursor.rowcount


def print_jobs_table(jobs: List[Job]):
    """Print jobs in a formatted table."""
    if not jobs:
        print("No jobs found.")
        return

    # Header
    print(f"{'JOB ID':<20} {'TYPE':<10} {'MODEL':<30} {'STATUS':<12} {'INSTANCE':<12} {'CREATED':<20}")
    print("-" * 110)

    for job in jobs:
        instance = str(job.instance_id) if job.instance_id else "-"
        created = job.created_at[:19] if job.created_at else "-"
        model = job.model_id[:28] + ".." if len(job.model_id) > 30 else job.model_id
        print(f"{job.job_id:<20} {job.job_type:<10} {model:<30} {job.status:<12} {instance:<12} {created:<20}")
