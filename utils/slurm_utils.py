"""
Helper functions to interact with SLURM
"""

import os

def submit_job(job_script, job_name, config_file):
    """
    Submit a job to SLURM.
    """
    os.system(f"sbatch {job_script} -J {job_name} -C {config_file}")


def cancel_job(job_id):
    """
    Cancel a job on SLURM.
    """
    os.system(f"scancel {job_id}")