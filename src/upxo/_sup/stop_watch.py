"""
stopwatch.py: A module containing the StopWatch class for measuring function execution time.

This module provides a class-based decorator named `StopWatch` that can be used to measure the execution time of any function and print the results in a user-friendly format, including minutes, seconds, and milliseconds.

Example usage:

```python
from stopwatch import StopWatch

@StopWatch()
def my_function():
    # Your function implementation here
    time.sleep(1)  # Simulate some processing

my_function()
"""

from functools import wraps
import time

class stopwatch:
    """A class-based decorator that measures function execution time and prints it in minutes, seconds, and milliseconds. This decorator can be used to measure the execution time of any function and print the results in a user-friendly format.

    Attributes:
        None
    Methods:
        __init__(self): Initializes the StopWatch object (no initialization required).
        __call__(self, func): The decorator implementation. This method is called when the StopWatch is used as a decorator.
    """

    def __init__(self):
        """Initializes the StopWatch object.

        This method is currently empty as no initialization is required for this
        example. However, it could be used to store additional information about the
        measured time or customize the output format in the future.
        """
        pass

    def __call__(self, func):
        """The decorator implementation. This method wraps the decorated function and measures its execution time. It then prints the results in minutes, seconds, and milliseconds.

        Args:
            func: The function to be decorated.
        Returns:
            The result of the decorated function.
        Raises:
            Exception: Any exception raised by the decorated function is re-raised
                to maintain the original error handling behavior.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            """The inner wrapper function that measures the execution time.
            Args:
                *args: Arguments passed to the decorated function.
                **kwargs: Keyword arguments passed to the decorated function.
            Returns:
                The result of the decorated function.
            Raises:
                Exception: Any exception raised by the decorated function is re-raised
                    to maintain the original error handling behavior.
            """
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                print(f"Error in function '{func.__name__}': {e}")
                raise  # Re-raise the exception to maintain the original error handling

            end_time = time.perf_counter()
            elapsed_time_seconds = end_time - start_time
            elapsed_time_milliseconds = elapsed_time_seconds * 1000
            elapsed_time_minutes = elapsed_time_seconds / 60

            print(
                f"Function '{func.__name__}' took {elapsed_time_minutes:.4f} minutes ({elapsed_time_seconds:.4f} seconds / {elapsed_time_milliseconds:.2f} milliseconds) to run."
            )

            return result

        return wrapper
