"""
Entry point. This is the ONLY file that wires concrete classes together
(Dependency Injection root) -- every other module depends on abstractions,
not on each other's concrete implementations. To swap a component (e.g. a
new reduction strategy, a different model, a different data source), change
the construction call here; no other file needs to change.
"""
import argparse

from config.config import ExperimentConfig, DimensionSchedule
from reduction.reduction_strategy import ReductionStrategyFactory
from models.model_wrapper import DistilBertWrapper
from metrics.registry import build_default_registry
from data.dataset_loader import HuggingFaceDatasetLoader
from pipeline.result_store import ResultStore
from pipeline.experiment_runner import ExperimentRunner
from visualization.plot_utils import plot_all_metrics
from evaluation.task_performance import LinearProbeAccuracyEvaluator
import os


def build_config_from_args(args) -> ExperimentConfig:
    schedule = DimensionSchedule(
        base_dim=args.base_dim,
        reduction_factor=args.reduction_factor,
        num_steps=args.num_steps,
    )
    return ExperimentConfig(
        model_name=args.model_name,
        seeds=[int(s) for s in args.seeds.split(",")],
        tasks=args.tasks.split(","),
        dimension_schedule=schedule,
        reduction_strategy_name=args.reduction_strategy,
        max_samples_for_metrics=args.max_samples,
        output_dir=args.output_dir,
        device=args.device,
    )


def main():
    parser = argparse.ArgumentParser(description="Phase 1: dimensional collapse characterization")
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--tasks", default="sst2")
    parser.add_argument("--seeds", default="13,42,2024")
    parser.add_argument("--base_dim", type=int, default=768)
    parser.add_argument("--reduction_factor", type=float, default=0.8)
    parser.add_argument("--num_steps", type=int, default=7)
    parser.add_argument("--reduction_strategy", default="truncation")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--no_task_performance", action="store_true",
        help="Skip the linear-probe task performance evaluation (faster, "
             "but task_performance will be null in results, same as before).",
    )
    parser.add_argument("--probe_cv_folds", type=int, default=3)
    args = parser.parse_args()

    config = build_config_from_args(args)

    # --- Dependency injection: construct every collaborator explicitly ---
    reduction_strategy = ReductionStrategyFactory.create(config.reduction_strategy_name)
    model_wrapper = DistilBertWrapper(
        model_name=config.model_name,
        reduction_strategy=reduction_strategy,
        device=config.device,
    )
    metric_registry = build_default_registry()
    dataset_loader = HuggingFaceDatasetLoader()

    # Namespace outputs by strategy so two strategies never collide/overwrite
    # each other's results -- this is what silently produced identical plots
    # for two supposedly different runs before.
    run_output_dir = os.path.join(config.output_dir, config.reduction_strategy_name)
    result_store = ResultStore(output_dir=run_output_dir)

    # BUGFIX: previously never constructed/passed -> task_performance was
    # always null. Wired in by default now; --no_task_performance opts out
    # if you want a faster metrics-only run.
    task_performance_evaluator = (
        None if args.no_task_performance
        else LinearProbeAccuracyEvaluator(cv_folds=args.probe_cv_folds)
    )

    runner = ExperimentRunner(
        config=config,
        model_wrapper=model_wrapper,
        metric_registry=metric_registry,
        dataset_loader=dataset_loader,
        result_store=result_store,
        task_performance_evaluator=task_performance_evaluator,
    )

    print(f"=== Phase 1 run ===")
    print(f"Reduction strategy : {config.reduction_strategy_name}")
    print(f"Dimensions         : {config.dimension_schedule.generate()}")
    print(f"Tasks              : {config.tasks}")
    print(f"Seeds              : {config.seeds}")
    print(f"Max samples/metric : {config.max_samples_for_metrics}")
    print(f"Task performance   : {'disabled' if task_performance_evaluator is None else 'linear probe, cv=' + str(args.probe_cv_folds)}")
    print(f"Output directory   : {run_output_dir}")
    print("=" * 20)
    result_store = runner.run()

    df = result_store.to_dataframe()
    metric_cols = [c for c in df.columns
                   if c not in ("task", "seed", "hidden_dim", "task_performance")]
    plot_paths = plot_all_metrics(df, metric_cols, output_dir=f"{run_output_dir}/plots")

    print(f"Saved results to {run_output_dir}/phase1_results.json")
    print(f"Saved {len(plot_paths)} plots to {run_output_dir}/plots/")


if __name__ == "__main__":
    main()