# afu

This repository implements AFU, the algorithm introduced in the paper:

 "AFU: Actor-Free critic Updates in off-policy RL for continuous control", Nicolas Perrin-Gilbert
 [https://arxiv.org/html/2404.16159v1](https://arxiv.org/html/2404.16159v1)

The code is based on a fork of [Rljax](https://github.com/toshikwa/rljax).

## INSTRUCTIONS

Install with:

    pip install -e .

Then run with:

    python simple_train.py --env [ENV] --algo [ALGO] --param [PARAM]

where:

* \[ENV\] can be "Ant", "HalfCheetah", "Hopper", "Humanoid", "InvertedDoublePendulum", "Reacher", or "Walker2d".
* \[ALGO\] can be "AFU-alpha", "AFU-beta", "SAC", "TD3", "IQL", "SQL", or "EQL".
* \[PARAM\] is a float between 0 and 1.

If \[ALGO\] is "AFU-alpha" or "AFU-beta", \[PARAM\] corresponds to $\varrho$ (see the [paper](https://arxiv.org/html/2404.16159v1)).

If \[ALGO\] is "IQL", \[PARAM\] corresponds to $\tau$, the expectile parameter of IQL (see [https://arxiv.org/abs/2110.06169](https://arxiv.org/abs/2110.06169)).

If \[ALGO\] is "SQL" or "EQL", \[PARAM\] corresponds to $\alpha$, the main parameter of SQL or EQL (see [https://arxiv.org/abs/2303.15810](https://arxiv.org/abs/2303.15810)).

Example of runs:

    python simple_train.py --env Ant --algo AFU-alpha --param 0.2

    python simple_train.py --env HalfCheetah --algo SAC
    
    python simple_train.py --env Hopper --algo TD3

    python simple_train.py --env InvertedDoublePendulum --algo IQL --param 0.9

    python simple_train.py --env Humanoid --algo AFU-beta --param 0.3
