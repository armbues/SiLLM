import os
import logging
from typing import Any, Dict, Optional, Union

import wandb
import numpy as np
import mlx.core as mx

logger = logging.getLogger("sillm")

class WandBLogger:
    """
    Weights & Biases logger for tracking training metrics and experiments.
    """
    def __init__(self,
                 project: str = "sillm",
                 name: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 tags: Optional[list] = None,
                 resume: bool = False,
                 id: Optional[str] = None
                 ):
        """
        Initialize WandB logger.
        
        Args:
            project: WandB project name
            name: Run name (optional)
            config: Run configuration/hyperparameters (optional)
            tags: Run tags (optional)
            resume: Whether to resume a previous run
            id: Run ID to resume (optional)
        """
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            tags=tags,
            resume=resume,
            id=id
        )
        
        self._step = 0
        logger.info(f"Initialized WandB logger - Project: {project}, Run: {self.run.name}")

    @property
    def step(self) -> int:
        """Current logging step."""
        return self._step

    def _convert_value(self, value: Any) -> Any:
        """Convert MLX arrays and other types to standard Python types."""
        if isinstance(value, mx.array):
            return value.item()
        elif isinstance(value, np.ndarray):
            return value.tolist()
        return value

    def log(self,
            metrics: Dict[str, Any],
            step: Optional[int] = None,
            commit: bool = True
            ):
        """
        Log metrics to WandB.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number (default: auto-increment)
            commit: Whether to commit the logs immediately
        """
        if step is not None:
            self._step = step
        
        # Convert any MLX arrays to Python types
        metrics = {k: self._convert_value(v) for k, v in metrics.items()}
        
        # Log metrics
        wandb.log(metrics, step=self._step, commit=commit)
        
        if commit:
            self._step += 1

    def log_hyperparams(self, params: Dict[str, Any]):
        """
        Log hyperparameters to WandB config.
        
        Args:
            params: Dictionary of hyperparameter names and values
        """
        # Convert any MLX arrays to Python types
        params = {k: self._convert_value(v) for k, v in params.items()}
        
        # Update wandb config
        wandb.config.update(params, allow_val_change=True)

    def log_model(self,
                  artifact_name: str,
                  path: str,
                  metadata: Optional[Dict[str, Any]] = None,
                  type: str = "model"
                  ):
        """
        Log model files/weights as a WandB artifact.
        
        Args:
            artifact_name: Name for the artifact
            path: Path to model file/directory
            metadata: Optional metadata to attach to artifact
            type: Artifact type (default: "model")
        """
        artifact = wandb.Artifact(
            name=artifact_name,
            type=type,
            metadata=metadata
        )
        
        if os.path.isfile(path):
            artifact.add_file(path)
        else:
            artifact.add_dir(path)
            
        self.run.log_artifact(artifact)

    def finish(self):
        """End the WandB run."""
        if self.run is not None:
            self.run.finish()
            self.run = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
