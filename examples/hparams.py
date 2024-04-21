from dataclasses import dataclass, field

from aitrainer import Hparams


@dataclass
class MLPHparams(Hparams):
    input_dim: int = field(default=784, metadata={'help': 'Dimensionality of the input.'})
    hidden_dim: int = field(default=128, metadata={'help': 'Dimensionality of the hidden layers.'})
    output_dim: int = field(default=10, metadata={'help': 'Dimensionality of the output.'})
    dropout: float = field(default=0.5, metadata={'help': 'Dropout probability.'})
    std: float = field(default=0.01, metadata={'help': 'Standard deviation of the weights.'})
