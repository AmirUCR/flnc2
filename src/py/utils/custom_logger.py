# src/py/utils/custom_logger.py
import sys
import logging

class bcolors:
    BLUE   = '\033[34m'
    GREEN  = '\033[32m'
    RED    = '\033[31m'
    ORANGE = '\033[38;5;208m'
    RESET  = '\033[0m'

LEVEL_COLORS = {
    logging.INFO:     bcolors.BLUE,
    logging.WARNING:  bcolors.ORANGE,
    logging.ERROR:    bcolors.RED,
    logging.CRITICAL: bcolors.RED,
}

class ColorFormatter(logging.Formatter):
    def __init__(self, *args, use_color=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_color = use_color

    def format(self, record):
        base = super().format(record)   # base is now "[timestamp] message"
        if not self.use_color:
            return f"> {base}"
        level_color = LEVEL_COLORS.get(record.levelno, bcolors.RESET)
        first, sep, rest = base.partition("\n")
        first = first.replace("Flnc2", f"{bcolors.BLUE}Flnc2{bcolors.RESET}")
        base = first + sep + rest

        return f"{level_color}>{bcolors.RESET} {base}"

def _log_uncaught(exc_type, exc, tb):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc, tb)   # keep Ctrl+C quiet and normal
        return
    logging.getLogger("flnc2").critical(
        "Unexpected error: %s", exc, exc_info=(exc_type, exc, tb)
    )

def set_quiet(quiet: bool = True, name: str = "flnc2") -> None:
    """Quiet: show only warnings and errors."""
    get_logger(name).setLevel(logging.WARNING if quiet else logging.INFO)
    
def get_logger(name="flnc2"):
    logger = logging.getLogger(name)
    if logger.handlers:          # already configured, don't add a second handler
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(ColorFormatter(
        fmt="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        use_color=sys.stderr.isatty(),
    ))
    logger.addHandler(handler)
    logger.propagate = False
    sys.excepthook = _log_uncaught

    return logger
    