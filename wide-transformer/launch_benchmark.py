import argparse

from omegaconf import OmegaConf
from mcli import RunConfig, create_run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # System args
    parser.add_argument("--cluster", type=str, default="r9z1")
    parser.add_argument("--ngpus", type=int, default=32)
    parser.add_argument("--device-batch-size", type=int, default=None)
    parser.add_argument("--autoresume", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[17])  # Add more later
    parser.add_argument("--local-debug", action="store_true")

    # Model args
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--n-layers", type=int, required=True)
    parser.add_argument("--expansion-ratio", type=int, required=True)
    parser.add_argument("--n-heads", type=int, default=16)

    args = parser.parse_args()

    run_base = f"profile-1B-dmodel-{args.d_model}-layers-{args.n_layers}-exp-{args.expansion_ratio}-heads-{args.n_heads}"

    for seed in args.seeds:
        base_run = RunConfig.from_file(
            f"wide-transformer/yamls/profiling/benchmark.yaml")

        run_name = f"{run_base}-sd-{seed}"
        base_run.name = run_name.lower()
        base_run.parameters["run_name"] = run_name

        base_run.parameters["model"]["d_model"] = args.d_model
        base_run.parameters["model"]["n_layers"] = args.n_layers
        base_run.parameters["model"]["expansion_ratio"] = args.expansion_ratio
        base_run.parameters["model"]["n_heads"] = args.n_heads

        base_run.cluster = args.cluster
        base_run.gpu_num = args.ngpus


        base_run.parameters["global_seed"] = seed

        base_run.parameters["loggers"]["wandb"]["tags"] = ["benchmark",
            "1B", f"dmodel-{args.d_model}", f"layers-{args.n_layers}",
            f"expansion-{args.expansion_ratio}", f"num_heads-{args.n_heads}"
        ]

        if args.cluster in ["r1z1", "r8z6"]:
            gpu_type = "a100_80gb"
            device_batch_size = 16
        elif args.cluster in ["r9z1"]:
            gpu_type = "h100_80gb"
            device_batch_size = 16
        else:
            gpu_type = "a100_40gb"
            device_batch_size = 4
        if args.device_batch_size is not None:
            device_batch_size = args.device_batch_size

        base_run.gpu_type = gpu_type
        base_run.parameters["device_train_microbatch_size"] = device_batch_size

        if args.local_debug:
            with open("debug.yaml", "w") as f:
                OmegaConf.save(config=OmegaConf.create(base_run.parameters),
                               f=f)
        else:
            run = create_run(base_run)
            print(
                f"Launched reference training for seed {seed} with in {run.name}"
            )
