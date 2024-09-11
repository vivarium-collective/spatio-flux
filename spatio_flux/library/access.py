import inspect

# Helper function to filter kwargs based on the function signature
def get_function_kwargs(func, all_kwargs):
    # Get the function signature
    sig = inspect.signature(func)
    # Filter kwargs to include only those arguments that are part of the function's signature
    return {k: v for k, v in all_kwargs.items() if k in sig.parameters}
