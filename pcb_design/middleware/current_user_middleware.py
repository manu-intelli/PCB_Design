from threading import local
from django.contrib.auth.models import AnonymousUser
from authentication.custom_authentication import CustomJWTAuthentication

_local = local()

class CurrentUserMiddleware:
    """
    Middleware to capture the current user from the request and store it in a thread-local variable.
    Supports both session-based and JWT-based authentication.
    """
    def __init__(self, get_response):
        self.get_response = get_response
        self.jwt_authenticator = CustomJWTAuthentication()

    def __call__(self, request):
        # Debug print for request path and current user state
        print(f"Processing request for path: {request.path}")

        # Check if the user is already authenticated via session
        if not request.user or request.user.is_anonymous:
            # Try JWT authentication using CustomJWTAuthentication
            try:
                user, _ = self.jwt_authenticator.authenticate(request)
                if user:
                    request.user = user
            except Exception:
                request.user = AnonymousUser()

        # Store the user in thread-local storage
        _local.user = request.user

        # Proceed with the request
        response = self.get_response(request)

        # Clean up after the request
        _local.user = None
        return response
