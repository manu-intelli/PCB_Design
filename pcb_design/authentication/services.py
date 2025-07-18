from .models import CustomUser
from .serializers import GetUserSerializer, UpdateUserSerializer, ForgotPasswordSerializer
from rest_framework.exceptions import ValidationError, NotFound
from . import authentication_logs


def get_users():
    try:
        users = CustomUser.objects.filter(is_active=True)
        if not users:
            return []
        serializer = GetUserSerializer(users, many=True)
        return serializer.data
    except Exception as ex:
        authentication_logs.error(f"Error: {ex}")
        raise ex


def update_user(data, pk):
    try:
        user = CustomUser.objects.get(pk=pk)
        serializer = UpdateUserSerializer(user, data=data)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return serializer.data
    except CustomUser.DoesNotExist as ex:
        authentication_logs.error(f"User with this id does not exist. {ex}")
        raise NotFound(ex)
    except ValidationError as e:
        authentication_logs.error(f"Validation Error: {e}")
        raise ValidationError(e)
    except Exception as ex:
        authentication_logs.error(f"Error: {ex}")
        raise ex



def delete_user(pk):
    try:
        user = CustomUser.objects.get(pk=pk)
        user.is_active = False
        user.save()
        return "User deleted successfully."
    except CustomUser.DoesNotExist as ex:                
        authentication_logs.error(f"User with this id does not exist. {ex}")
        raise NotFound(ex)
    except Exception as ex:
        authentication_logs.error(f"Error: {ex}")
        raise ex
    

def reset_user_password(data):
    try:
        user = CustomUser.objects.get(email=data.get('email'))        
        serializer = ForgotPasswordSerializer(data=data)        
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return serializer
    
    except CustomUser.DoesNotExist as e:
        authentication_logs.error(f"User with this email does not exist. {e}")
        raise NotFound("User with this email does not exist. {}".format(e))
    except ValidationError as e:
        authentication_logs.error(f"Validation Error: {e}")
        raise ValidationError(e)
    except Exception as ex:
        authentication_logs.error(f"Error: {ex}")
        raise ex