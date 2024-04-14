from collections.abc import Iterator

from deepspeed import DeepSpeedEngine
from torch import nn

from aitrainer.hparams import Hparams
from aitrainer.utils import to_device


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, std):
        """Multi-layer perceptron.

        Args:
            input_dim: Dimensionality of the input.
            hidden_dim: Dimensionality of the hidden layers.
            output_dim: Dimensionality of the output.
            dropout: Dropout probability.
            std: Standard deviation of the weights.

        Returns:
            Initialized MLP."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            std,
        )

    def forward(self, x):
        x = self.net(x)
        return x


def train_step_fn(  # pylint: disable=unused-argument
    engine_dict: dict[str, DeepSpeedEngine],
    dl_iter: Iterator,
    hparams: Hparams,
    step: int,
) -> tuple[float, dict]:
    """Train step function for the MLP model"""
    model = engine_dict['model']
    batch = to_device(next(dl_iter), model.device)

    # Forward pass
    loss = model(batch['inputs'])

    # Backward pass
    model.backward(loss)

    # Weight update
    model.step()

    return loss.item(), {}
