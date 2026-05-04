import logging
from argparse import ArgumentParser

from .unpack_ms import create_measurement_sets

logger = logging.getLogger(__name__)


def parse_args(args):
    parser = ArgumentParser()
    parser.add_argument("input_root")
    parser.add_argument("output_root")
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args(args)


def logger_callback(message):
    logger.info(message)


def main(args=None):
    parsed_args = parse_args(args)
    create_measurement_sets(
        input_root=parsed_args.input_root,
        output_root=parsed_args.output_root,
        overwrite=parsed_args.overwrite,
        logger_fn=logger_callback,
    )
