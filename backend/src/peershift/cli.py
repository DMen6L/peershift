"""Command line entry point."""
import argparse
from .simulation import Config, run


def main():
    parser = argparse.ArgumentParser(prog='peershift')
    commands = parser.add_subparsers(dest='command', required=True)
    simulate = commands.add_parser('simulate')
    simulate.add_argument('--config', required=True)
    simulate.add_argument('--output', required=True)
    train = commands.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--download", action="store_true")
    mpi_demo = commands.add_parser("mpi-demo", help="Exchange known models across MPI ranks without training")
    mpi_demo.add_argument("--config", required=True)
    mpi_demo.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "mpi-demo":
            try:
                from .mpi_demo import run_demo
                from mpi4py import MPI
            except (ImportError, RuntimeError) as error:
                parser.error(f"MPI requires a compatible MPI runtime and: uv sync --project backend --extra mpi ({error})")
            run_demo(args.config, args.output)
        elif args.command == "train":
            try:
                from .training import TrainingConfig, run_training
            except ImportError:
                parser.error("Install learning dependencies: uv sync --project backend --extra learning")
            run_training(TrainingConfig.load(args.config), args.output, download=args.download)
        else:
            run(Config.load(args.config), args.output)
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        parser.error(str(error))
