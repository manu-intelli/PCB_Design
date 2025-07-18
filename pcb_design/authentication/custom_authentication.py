from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.exceptions import AuthenticationFailed
from authentication.models import CustomUser

class CustomJWTAuthentication(JWTAuthentication):
    def authenticate(self, request):          
        if '/swagger/' in request.path:
            return None

        # Safely handle the return value from super().authenticate()
        result = super().authenticate(request)
        if result is None:
            return None

        user, token = result

        if user and getattr(user, 'is_logged_out', False):
            raise AuthenticationFailed("User is logged out")

        return user, token
