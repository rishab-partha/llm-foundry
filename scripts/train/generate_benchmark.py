import contextlib
import os
import sys
import torch
import warnings

from composer import Trainer
from composer.core import Evaluator
from composer.utils import dist, get_device, reproducibility
from composer.callbacks import MemoryMonitor
sys.path.insert(1, '/composer/tests/datasets')
from code_eval_inputs import get_code_eval_inputs
from omegaconf import OmegaConf as om
from composer.loggers import WandBLogger
from composer.core import Precision, get_precision_context
import time

from llmfoundry import (COMPOSER_MODEL_REGISTRY, build_finetuning_dataloader,
                        build_text_denoising_dataloader)
from llmfoundry.data.text_data import build_text_dataloader
from llmfoundry.models.utils import init_empty_weights
from llmfoundry.utils.builders import (build_algorithm, build_callback,
                                       build_icl_evaluators, build_logger,
                                       build_optimizer, build_scheduler,
                                       build_tokenizer)
from llmfoundry.utils.config_utils import log_config, update_batch_size_info



def main(cfg):
    torch.cuda.empty_cache()
    fsdp_config = cfg.get('fsdp_config', None)
    fsdp_config = om.to_container(fsdp_config,
                                  resolve=True) if fsdp_config else None
    dist.initialize_dist(get_device(None), timeout=80.0)
    if dist.get_world_size() == 1 and fsdp_config is not None:
        warnings.warn(
            'FSDP is not applicable for single-GPU training. Reverting to DDP.')
        cfg.pop('fsdp_config')
        fsdp_config = None
    
    eval_dataloader = get_code_eval_inputs(cfg.tokenizer.name, cfg.device_eval_batch_size)
    tokenizer = build_tokenizer(cfg.tokenizer)
    trainer = Trainer(
            run_name='benchmark_generate_fsdp',
            model=COMPOSER_MODEL_REGISTRY[cfg.model.name](cfg.model, tokenizer),
            fsdp_config=fsdp_config,  # type: ignore
            precision=cfg.precision,
            dist_timeout = 80.0,
            load_weights_only=True,
            eval_dataloader = Evaluator(label = 'metric', dataloader=eval_dataloader, metric_names = ['InContextLearningCodeEvalAccuracy']),
            loggers = [WandBLogger(project='rishab-mosaicml-tests', entity='mosaic-ml')],
            callbacks = [MemoryMonitor()],
        )

    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print("Before eval")
    start_time = time.time()
    trainer.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()
    
    tot_time = end_time - start_time

    print(f"Time elapsed: {tot_time} seconds")
if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    main(cfg)
