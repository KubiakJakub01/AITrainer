from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Hparams:
    # Data
    train_data_dir: Path = field(
        default=Path('data/train'),
        metadata={'help': 'Directory for data'},
    )
    valid_data_dir: Path = field(
        default=Path('data/valid'),
        metadata={'help': 'Directory for data'},
    )
    log_dir: Path = field(
        default=Path('models/log'),
        metadata={'help': 'Directory for logs'},
    )
    checkpoint_dir: Path = field(
        default=Path('models/checkpoints'),
        metadata={'help': 'Directory for checkpoints'},
    )
    loader_num_workers: int = field(
        default=4,
        metadata={'help': 'Number of workers for data loader'},
    )
    batch_size: int = field(
        default=4,
        metadata={'help': 'Batch size'},
    )

    # Training
    total_steps: int = field(
        default=100000,
        metadata={'help': 'Total number of steps to train for'},
    )
    log_every: int = field(
        default=100,
        metadata={'help': 'Number of steps between logs'},
    )
    base_checkpoint: int | None = field(
        default=None,
        metadata={'help': 'Path to base checkpoint. If None, start from scratch'},
    )
    steps_per_log: int = field(
        default=100,
        metadata={'help': 'Number of steps between logs'},
    )
    steps_per_ckpt: int = field(
        default=1000,
        metadata={'help': 'Number of steps between validation runs'},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={'help': 'Number of steps to accumulate gradients over'},
    )
    gradient_clipping: float = field(
        default=1.0,
        metadata={'help': 'Gradient clipping value'},
    )

    def __post_init__(self):
        self.train_data_dir = Path(self.train_data_dir)
        self.valid_data_dir = Path(self.valid_data_dir)
        self.log_dir = Path(self.log_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def deepspeed_config(self):
        return {}
