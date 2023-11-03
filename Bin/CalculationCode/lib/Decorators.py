import timeit


def timer(func):
    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        result = func(*args, **kwargs)
        end = timeit.default_timer()
        print(f"\n{func.__name__} took {end - start:.2f} sec.\n")
        return result
    return wrapper
