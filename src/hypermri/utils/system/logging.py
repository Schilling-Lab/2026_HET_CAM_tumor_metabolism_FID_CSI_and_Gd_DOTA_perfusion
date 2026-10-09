from dataclasses import dataclass


@dataclass
class log_modes_generator:
    critical: int = 50  # logging.CRITICAL
    error: int = 40  # logging.ERROR
    warning: int = 30  # logging.WARNING
    info: int = 20  # logging.INFO
    debug: int = 10  # logging.DEBUG
    notset: int = 0  # logging.NOTSET

    def __getitem__(cls, key):
        if isinstance(key, int):
            return key
        if isinstance(key, str):
            return getattr(cls, key.lower())
        raise ValueError(f"'{key}' is not a valid logging mode.")

    def get(self, key, default=None):
        try:
            return self[key]
        except (AttributeError, ValueError):
            return default


LOG_MODES = log_modes_generator()


def init_default_logger(
    name, default_mode="error", fstring="[%(levelname)s] %(name)s: %(message)s "
):
    """Initialize the default logger class.

    fstring:
        Format of the log message, example: '%(asctime)s:%(name)s:%(message)s '
    """
    import logging

    logger = logging.getLogger(name)

    logger.setLevel(LOG_MODES[default_mode])

    formatter = logging.Formatter(fstring)
    # formatter = logging.Formatter()

    # Multiple handlers (to stream and/or to file) are possible.
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # we have to ensure we don't add a handler every time the class is called.
    if not logger.hasHandlers():
        logger.addHandler(stream_handler)

    return logger


def set_hypermri_log_level(log_mode):
    """Set log level for all hypermri loggers and ensure they have handlers.

    This function finds all active loggers in the hypermri namespace,
    sets their level to the specified log_mode, and ensures they have
    proper handlers to display messages.

    Parameters
    ----------
    log_mode : str or int
        The log level to set (e.g., 'debug', 'info', 'warning', 'error')
    """
    import logging

    # Convert log_mode to level if it's a string
    if isinstance(log_mode, str):
        level = LOG_MODES[log_mode]
    else:
        level = log_mode

    # Get all loggers and filter for hypermri ones
    all_loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    hypermri_loggers = [
        logger for logger in all_loggers if logger.name.startswith("hypermri")
    ]

    # Set level for all hypermri loggers and ensure they have handlers
    for logger in hypermri_loggers:
        logger.setLevel(level)

        # If logger has no handlers, add a default one
        if not logger.hasHandlers():
            # Create a handler with the same format as init_default_logger
            formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s ")
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

            # Set the handler level to match the logger level
            stream_handler.setLevel(level)


def underoverline(text):
    """Overline and underline a string with dashes and print."""
    line = "-" * len(text)
    print(f"{line}\n{text}\n{line}")


def debug_log(
    message,
    file_name="/tmp/tmp_debug.log",
    use_timestamp=True,
    write_to_file=False,
    print_error=False,
    numpy_full_print=False,
):
    """
    Simple debug logging function that writes timestamped messages to a temporary file.
    Parameters
    ----------
    message : str
        The message to log
    file_name : str, optional
        The file to write the log message to (default is '/tmp/tmp_debug.log')
    use_timestamp : bool, optional
        Whether to prepend a timestamp to the message (default is True)
    write_to_file : bool, optional
        Whether to write the message to the file (default is True)
    print_error : bool, optional
        Whether to print an error message if writing fails (default is False)
    numpy_full_print : bool, optional
        Whether to enable full NumPy array printing (default is False)
    """
    if not write_to_file:
        return

    try:
        import datetime

        import numpy as np

        # Enable full NumPy printing if requested
        if numpy_full_print:
            np.set_printoptions(threshold=np.inf, linewidth=np.inf, suppress=True)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with open(file_name, "a") as f:
            if use_timestamp:
                f.write(f"[{timestamp}] {message}\n")
            else:
                f.write(f"{message}\n")
            f.flush()
    except Exception as e:
        if print_error:
            print(f"Failed to write debug log: {e}")
        else:
            pass
