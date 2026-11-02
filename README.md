# SEMEX

Implementation of **Semantic-level Exploration for Multi-Agent Reinforcement Learning**, built on [PyMARL](https://github.com/oxwhirl/pymarl).

SEMEX improves multi-agent exploration through two components:

- **Semantic Quantizer (SQ):** uses VQ-VAE to encode observation-action pairs into discrete semantic prototypes.
- **Sliding Window Counting (SWC):** provides count-based exploration bonuses in semantic space, sharing exploration statistics across semantically similar situations.

This repository integrates SEMEX with QMIX for experiments on the [StarCraft Multi-Agent Challenge (SMAC)](https://github.com/oxwhirl/smac).

## Installation

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Install StarCraft II and the SMAC maps, and set `SC2PATH` to your StarCraft II installation directory. Keep the StarCraft II version consistent when comparing results.

## Run

From the project root:

```bash
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=2s3z
```

Configuration files are in `src/config/`. Change `env_args.map_name` to run another map. Experiment outputs are saved in `results/`.

## Acknowledgements

This project builds on PyMARL and SMAC. Code is licensed under [Apache License 2.0](LICENSE).
