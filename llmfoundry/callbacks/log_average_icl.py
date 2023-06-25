# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Periodically log generations to wandb from a set of prompts."""
from typing import List, Union
import copy
import re

import torch
import wandb
from composer.core import Callback, State, get_precision_context
from composer.loggers import Logger, WandBLogger
from composer.utils import dist, ensure_tuple
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


class AverageICLLogger(Callback):

    def __init__(self, num_shots: List[int]):
        """Averages different icl task performance for same # shots.

        Args: 
         - num_shots (List[int]): The list of num shots being used for ICL
        """
        self.num_shots = num_shots
        self.wandb_logger = None

    def init(self, state: State, logger: Logger):
        if dist.get_global_rank() == 0:
            for destination in ensure_tuple(logger.destinations):
                if isinstance(destination, WandBLogger):
                    self.wandb_logger = destination

    def eval_after_all(self, state: State, logger: Logger):
        eval_metrics = copy.deepcopy(state.eval_metrics)
        num_shot_avgs = {num_shot: [] for num_shot in self.num_shots}
        for _, metrics in eval_metrics.items():
            for metric_name, metric_val in metrics.items():
                match = re.search(r"(\d+)-shot", metric_name)
                if not match:
                    continue
                num_shots = int(match.group(1))
                num_shot_avgs[num_shots].append(metric_val.compute())
        num_shot_avgs = {
            f"metrics/icl/{num_shot}-shot/avg": sum(perfs) / len(perfs)
            for num_shot, perfs in num_shot_avgs.items()
        }
