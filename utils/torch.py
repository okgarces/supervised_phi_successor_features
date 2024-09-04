from typing import Iterable

import numpy as np

import torch


def polyak_update(params: Iterable[torch.nn.Parameter], target_params: Iterable[torch.nn.Parameter], tau: float) -> None:
    with torch.no_grad():
        for param, target_param in zip(params, target_params):
            if tau == 1:
                target_param.data.copy_(param.data)
            else:
                target_param.data.mul_(1.0 - tau)
                torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)


def linearly_decaying_epsilon(initial_epsilon, decay_period, step, warmup_steps, final_epsilon):
    """Returns the current epsilon for the agent's epsilon-greedy policy.
    This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
    al., 2015). The schedule is as follows:
    Begin at 1. until warmup_steps steps have been taken; then
    Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
    Use epsilon from there on.
    Args:
    decay_period: float, the period over which epsilon is decayed.
    step: int, the number of training steps completed so far.
    warmup_steps: int, the number of steps taken before epsilon is decayed.
    epsilon: float, the final value to which to decay the epsilon parameter.
    Returns:
    A float, the current epsilon value computed according to the schedule.
    """
    steps_left = decay_period + warmup_steps - step
    bonus = (initial_epsilon - final_epsilon) * steps_left / decay_period
    bonus = np.clip(bonus, 0., 1. - final_epsilon)
    return final_epsilon + bonus