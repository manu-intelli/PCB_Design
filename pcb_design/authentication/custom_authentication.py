from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.exceptions import AuthenticationFailed
import jwt
from django.conf import settings
from authentication.models import CustomUser


class CustomJWTAuthentication(JWTAuthentication):
    def authenticate(self, request):          
        if '/swagger/' in request.path:            
            return None
    
        user, token = super().authenticate(request)                
        if user and user.is_logged_out:
            raise AuthenticationFailed("User is logged out")

        return user, token
