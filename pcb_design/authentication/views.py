from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import authenticate
from django.contrib.auth.models import Group
from rest_framework import status
from rest_framework.exceptions import ValidationError, NotFound
from .custom_permissions import IsAuthorized
from .serializers import RegisterSerializer,RoleSerializer, UpdateUserSerializer
from .models import CustomUser
from .services import reset_user_password, get_users, update_user, delete_user
from . import authentication_logs


class UserRegistrationView(APIView):
    permission_classes = [IsAuthorized]

    def post(self, request):
        authentication_logs.info(f"User Registration Request: {request.data}")
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            authentication_logs.info(f"User Registration Successful: {user}")
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        authentication_logs.error(f"User Registration Failed: {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



class LoginView(APIView):
    permission_classes = [] 
    authentication_classes = [] 

    @swagger_auto_schema(
        operation_description="User login with email and password",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            required=['email', 'password'],
            properties={
                'email': openapi.Schema(type=openapi.TYPE_STRING, format='email', example="user@example.com"),
                'password': openapi.Schema(type=openapi.TYPE_STRING, format='password', example="securepassword123")
            },
        ),
        responses={
            200: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'refresh': openapi.Schema(type=openapi.TYPE_STRING, description="Refresh token"),
                    'access': openapi.Schema(type=openapi.TYPE_STRING, description="Access token"),
                    'role': openapi.Schema(type=openapi.TYPE_ARRAY, items=openapi.Items(type=openapi.TYPE_STRING), description="List of user roles"),
                    'full_name': openapi.Schema(type=openapi.TYPE_STRING, description="Full name of the user"),
                }
            ),
            401: openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'error': openapi.Schema(type=openapi.TYPE_STRING, example="Invalid credentials"),
                }
            )
        }
    )

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


class UserListAPIView(APIView):
    permission_classes = [IsAuthorized]
    http_method_names = ['get']  # ✅ Only allow GET

    @swagger_auto_schema(
        operation_description="Retrieve a list of all users",
        responses={
            200: openapi.Response("Success", UpdateUserSerializer(many=True)),
            500: "Internal Server Error"
        }
    )
    def get(self, request):
        try:
            authentication_logs.info(f"Getting all users -- requested by {request.user.email}")
            response = get_users()  # 🔁 Should return serialized data
            return Response(response, status=status.HTTP_200_OK)
        except Exception as e:
            authentication_logs.error(f"Exception occurred while getting users: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class UserAPIView(APIView):
    permission_classes = [IsAuthorized]

    @swagger_auto_schema(
        operation_description="Get a specific user by ID",
        responses={
            200: openapi.Response("User Retrieved", UpdateUserSerializer),
            404: "User Not Found",
            500: "Internal Server Error"
        }
    )
    def get(self, request, pk):
        try:
            authentication_logs.info(f"Getting the User -- requested by {request.user.email} for user id: {pk}")
            user = CustomUser.objects.get(pk=pk)
            serializer = UpdateUserSerializer(user)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except CustomUser.DoesNotExist:
            authentication_logs.error(f"User with id {pk} not found.")
            return Response({"error": f"User with id {pk} not found."}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            authentication_logs.error(f"Exception while retrieving user: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @swagger_auto_schema(
        operation_description="Update a user by ID. This will replace their full name and roles.",
        request_body=UpdateUserSerializer,
        responses={
            200: openapi.Response("User Updated", UpdateUserSerializer),
            400: "Validation Error",
            404: "User Not Found",
            500: "Internal Server Error"
        }
    )
    def put(self, request, pk):
        try:
            authentication_logs.info(f"Updating the User -- request done by {request.user.email} for user id: {pk}")
            response = update_user(request.data, pk)
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

    @swagger_auto_schema(
        operation_description="Delete a user by ID (soft delete by deactivating them)",
        responses={
            200: openapi.Response("User Deleted", openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    "message": openapi.Schema(type=openapi.TYPE_STRING, example="User deleted successfully.")
                }
            )),
            404: "User Not Found",
            500: "Internal Server Error"
        }
    )
    def delete(self, request, pk):
        try:
            authentication_logs.info(f"Deleting the User -- request done by {request.user.email} for user id: {pk}")
            response = delete_user(pk)
            authentication_logs.info(f"Deleted the User: {response}")
            return Response({"message": response}, status=status.HTTP_200_OK)
        except NotFound as e:
            authentication_logs.error(f"User Not Found: {e} for user id: {pk}")
            return Response({"error": f"User Not Found: {e}"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            authentication_logs.error(f"Exception occurred While Deleting the User: {e} for user id: {pk}")
            return Response({"error": f"Exception occurred While Deleting the User: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class RoleListView(APIView):
    permission_classes = [IsAuthorized]
    """
    API endpoint to list all available roles.
    """
    def get(self, request):
        roles = Group.objects.exclude(name="Admin")
        serializer = RoleSerializer(roles, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)