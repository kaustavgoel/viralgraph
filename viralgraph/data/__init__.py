"""Data ingestion, synthetic network generation, and empirical dataset loading."""

from viralgraph.data.synthetic_generator import (
    generate_barabasi_albert,
    generate_erdos_renyi,
    generate_stochastic_block_model,
    generate_watts_strogatz,
)

__all__ = [
    "generate_barabasi_albert",
    "generate_erdos_renyi",
    "generate_watts_strogatz",
    "generate_stochastic_block_model",
]
