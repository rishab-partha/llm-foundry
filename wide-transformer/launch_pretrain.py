import argparse

from omegaconf import OmegaConf
from mcli import RunConfig, create_run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # System args
    parser.add_argument("--cluster", type=str, default="r7z2")
    parser.add_argument("--ngpus", type=int, default=32)
    parser.add_argument("--autoresume", action="store_true")
    parser.add_argument("--preemptible", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[17])  # Add more later
    parser.add_argument("--local-debug", action="store_true")

    # Model args
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--n-layers", type=int, required=True)
    parser.add_argument("--expansion-ratio", type=int, required=True)

    args = parser.parse_args()

    run_base = f"1B-dmodel-{args.d_model}-layers-{args.n_layers}-expansion-{args.expansion_ratio}"

    for seed in args.seeds:
        base_run = RunConfig.from_file(
            f"wide-transformer/yamls/pretrain/base.yaml")

        run_name = f"{run_base}-sd-{seed}"
        base_run.name = run_name.lower()
        base_run.parameters["run_name"] = run_name

        base_run.parameters["model"]["d_model"] = args.d_model
        base_run.parameters["model"]["n_layers"] = args.n_layers
        base_run.parameters["model"]["expansion_ratio"] = args.expansion_ratio

        base_run.cluster = args.cluster
        base_run.gpu_num = args.ngpus

        if args.preemptible:
            base_run.scheduling = {"resumable": True, "priority": "low"}
            base_run.parameters["autoresume"] = True
        else:
            base_run.parameters["autoresume"] = args.autoresume

        base_run.parameters["global_seed"] = seed

        base_run.parameters["loggers"]["wandb"]["tags"] = [
            "1B", "pile", f"dmodel-{args.d_model}", f"layers-{args.n_layers}",
            f"expansion-{args.expansion_ratio}"
        ]

        if args.cluster in ["r1z1", "r8z6"]:
            gpu_type = "a100_80gb"
            device_batch_size = 16
        else:
            gpu_type = "a100_40gb"
            device_batch_size = 8

        base_run.gpu_type = gpu_type
        base_run.parameters["device_train_microbatch_size"] = device_batch_size
        base_run.parameters["device_eval_batch_size"] = device_batch_size

        if args.local_debug:
            with open("debug.yaml", "w") as f:
                OmegaConf.save(config=OmegaConf.create(base_run.parameters),
                               f=f)
        else:
            run = create_run(base_run)
            print(
                f"Launched reference training for seed {seed} with in {run.name}"
            )
