"""Command line entry point."""
import argparse
from .simulation import Config, run


def main():
    parser = argparse.ArgumentParser(prog='peershift')
    commands = parser.add_subparsers(dest='command', required=True)
    simulate = commands.add_parser('simulate')
    simulate.add_argument('--config', required=True)
    simulate.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        run(Config.load(args.config), args.output)
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))
