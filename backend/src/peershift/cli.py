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
    args = parser.parse_args()
    try:
        if args.command == "train":
            try:
                from .training import TrainingConfig, run_training
            except ImportError:
                parser.error("Install learning dependencies: uv sync --project backend --extra learning")
            run_training(TrainingConfig.load(args.config), args.output, download=args.download)
        else:
            run(Config.load(args.config), args.output)
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        parser.error(str(error))
