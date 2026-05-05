import sys
import logging
import traceback

def error_message_detail(error_message):
    exc_type, exc_value, exc_tb = sys.exc_info()
    if exc_tb is not None:
        tb = traceback.extract_tb(exc_tb)
        if tb:
            # Extract the last frame (where the actual error happened)
            last_call = tb[-1]
            file_name = last_call.filename
            line_number = last_call.lineno
            return f"Error occurred in python script name [{file_name}] line number [{line_number}] error message [{error_message}]"
    return str(error_message)

class MyException(Exception):
    def __init__(self, error_message):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message)
        logging.error(self.error_message)

    def __str__(self) -> str:
        """
        Returns the string representation of the error message.
        """
        return self.error_message