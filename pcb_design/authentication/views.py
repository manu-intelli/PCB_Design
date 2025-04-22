from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import authenticate
from rest_framework import status
from rest_framework.exceptions import ValidationError, NotFound
from .custom_permissions import IsAuthorized
from .serializers import RegisterSerializer
from .services import reset_user_password, get_users, update_user, delete_user
from . import authentication_logs


class UserRegistrationView(APIView):
    permission_classes = [IsAuthorized]
    def post(self, request):
        authentication_logs.info(f"User Registration Request: {request.data}")
        role = request.data.get('role', 'CADesigner')
        authentication_logs.info(f"Role: {role}")
        serializer = RegisterSerializer(data=request.data, context={'role': role})
        if serializer.is_valid():
            user = serializer.save()
            authentication_logs.info(f"User Registration Successful: {user}")
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        authentication_logs.error(f"User Registration Failed: {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LoginView(APIView):
    permission_classes = [] 
    authentication_classes = [] 

    def post(self, request):        
        email = request.data.get('email')
        authentication_logs.info(f"Login Request: {email}")
        password = request.data.get('password')
        user = authenticate(email=email, password=password)                
        if user:                
            # Retrieve the user's group names (roles)
            user_roles = [group.name for group in user.groups.all()]
            authentication_logs.info(f"Login Successful: {email} -- {user_roles}")
            user.is_logged_out = False
            user.save()
            refresh = RefreshToken.for_user(user)
            return Response({
                'refresh': str(refresh),
                'access': str(refresh.access_token),
                'role': user_roles,
                'full_name': user.full_name
            })
        authentication_logs.error(f"Login Failed: {email} -- Invalid credentials -- {user}")
        return Response({'error': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)


class LogoutView(APIView):
    def post(self, request):
        try:             
            user = request.user            
            authentication_logs.info(f"Logout Request: {user.email}")
            user.is_logged_out = True
            user.save()            
            authentication_logs.info(f"Logout Successful: {user.email}")
            return Response({'message': 'Logout successful'}, status=status.HTTP_200_OK)
        except Exception as e:
            authentication_logs.error(f"Logout Failed: {e} for {user.email}")
            return Response({'error': 'Invalid token'}, status=status.HTTP_400_BAD_REQUEST)

class ForgetPasswordView(APIView):
    permission_classes = [] 
    authentication_classes = []
    def post(self,request):
        try:
            authentication_logs.info(f"Forgot Password Request: {request.data.get('email')}")
            data= request.data            
            response = reset_user_password(data)            
            authentication_logs.info(f"Updated the Password for {request.data.get('email')}")
            return Response(response.data, status=status.HTTP_200_OK)
        
        except ValidationError as e:
            authentication_logs.error(f"Validation Error: {e} for {request.data.get('email')}")
            return Response({"error": f"Validation Error: {e}"}, status=status.HTTP_400_BAD_REQUEST)
        except NotFound as e:
            authentication_logs.error(f"User Not Found: {e} for {request.data.get('email')}")
            return Response({"error": f"User Not Found: {e}"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            authentication_logs.error(f"Exception occurred: {e} for {request.data.get('email')}")
            return Response({"error": f"Exception occurred: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class UserAPIView(APIView):
    permission_classes = [IsAuthorized]
    def get(self, request):
        try:
            authentication_logs.info(f"Getting the Users -- request done by {request.user.email}")
            response = get_users()
            authentication_logs.info(f"Users: {response}")
            return Response(response, status=status.HTTP_200_OK)
        except Exception as e:
            authentication_logs.error(f"Exception occurred While Getting the Users: {e}")
            return Response({"error": f"Exception occurred While Getting the Users: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)    
    
    def put(self, request, pk):
        try:
            authentication_logs.info(f"Updating the User -- request done by {request.user.email} for user id: {pk}")
            response = update_user(request.data,pk)
            authentication_logs.info(f"Updated the User: {response}")
            return Response(response, status=status.HTTP_200_OK)
        except NotFound as e:
            authentication_logs.error(f"User Not Found: {e} for user id: {pk}")
            return Response({"error": f"User Not Found: {e}"}, status=status.HTTP_404_NOT_FOUND)
        except ValidationError as e:
            authentication_logs.error(f"Validation Error: {e} for user id: {pk}")
            return Response({"error": f"Validation Error: {e}"}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            authentication_logs.error(f"Exception occurred While Updating the User: {e} for user id: {pk}")
            return Response({"error": f"Exception occurred While Updating the User: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def delete(self, request, pk):
        try:    
            authentication_logs.info(f"Deleting the User -- request done by {request.user.email} for user id: {pk}")    
            response = delete_user(pk)
            authentication_logs.info(f"Deleted the User: {response}")
            return Response(response)
        except NotFound as e:
            authentication_logs.error(f"User Not Found: {e} for user id: {pk}")
            return Response({"error": f"User Not Found: {e}"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            authentication_logs.error(f"Exception occurred While Deleting the User: {e} for user id: {pk}")
            return Response({"error": f"Exception occurred While Deleting the User: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
