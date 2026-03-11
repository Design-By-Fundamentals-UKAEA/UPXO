import tqdm
import time

def with_progress_bar(func):
    """Decorator to add a tqdm progress bar to a function."""
    def wrapper(*args, **kwargs):
        if 'total' in kwargs:  # Check if 'total' is provided
            total = kwargs.pop('total')
            pbar = tqdm.tqdm(total=total)
        else:
            pbar = tqdm.tqdm()  # No 'total' -> indeterminate mode

        for i in range(total if 'total' in kwargs else 10):  # Flexible iteration
            pbar.set_description(f'Processing step {i+1}')
            result = func(*args, **kwargs)  # Call the original function
            pbar.update(1)
        pbar.close()  # Close the progress bar
        return result
    return wrapper



@with_progress_bar
def my_computation(total=None):
    # Your computation here (you can remove time.sleep if needed)
    time.sleep(0.1)

# Example 1: With total iterations
my_computation(total=50)

# Example 2: Without total iterations
my_computation()
