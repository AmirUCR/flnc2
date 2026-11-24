import os
import sys
from pathlib import Path

from src.py.utils.custom_logger import get_logger
from src.py.utils.exceptions import StepError
from src.py.utils.configurator import Configurator
from src.py.pipelines import (run_training, run_inference, 
                              run_reads_pipeline, run_bed_pipeline)

def main() -> int:
    try:
        logger = get_logger()
        config = Configurator()

        # Read command line arguments and config.yaml.
        # Some arguments are not shown in config.yaml and are defaulted in parse_configurations().
        # Arguments specified on the command line have priority and will replace those in config.yaml.
        config.parse_configurations()

        if "--soundcheck" not in sys.argv:
            config.begruessung()

        # This is the only function that changes the arguments in the config file.
        # Warnings and info to help with user error. Also hard-codes some paths and arguments.
        args = config.check_and_fix_configurations()
        
        # Transforms all options already defined in config.parse_configurations
        # to FLNC2_{yaml_option_but_in_upper_case}. These will be 
        # environment variables used by the scripts under directory bash_scripts.
        # If an option is not defined in config.parse_configurations,
        # but present in config.yaml, it will be ignored.
        env_vars = config.build_env(args)

        if args.mode == "train":
            run_training(args)
        else:
            features = run_bed_pipeline(args) if args.bed not in (None, "") else run_reads_pipeline(args)
            run_inference(features, args)
    except StepError as e:
        logger.error("Flnc2 aborted: %s", e)
        return 1
    
    logger.info("Flnc2 is done.")
    return 0

if __name__ == '__main__':
    sys.exit(main())
    