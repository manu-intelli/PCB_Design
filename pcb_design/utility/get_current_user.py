from middleware.current_user_middleware import _local
def get_current_user():
    """
    Utility function to retrieve the current user from thread-local storage.
    """
    return getattr(_local, 'user', None)
