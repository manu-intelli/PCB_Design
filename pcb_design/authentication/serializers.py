from rest_framework import serializers
from django.contrib.auth.models import Group

from .models import CustomUser
from rest_framework import serializers
from rest_framework.validators import UniqueValidator
from django.contrib.auth.password_validation import validate_password

valid_roles = ['Admin', 'CADesigner', 'Approver', 'Verifier']
from . import authentication_logs


class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)
    class Meta:
        model = CustomUser
        fields = ['id','email', 'password']
    
    def create(self, validated_data):
        return CustomUser.objects.create_user(**validated_data)

 
class RegisterSerializer(serializers.ModelSerializer):
    email = serializers.EmailField(
        required=True,
        validators=[UniqueValidator(queryset=CustomUser.objects.all())]
    )
    password = serializers.CharField(write_only=True, required=True, validators=[validate_password])
    password2 = serializers.CharField(write_only=True, required=True)
    full_name = serializers.CharField(required=True) 
  
    class Meta:
        model = CustomUser
        fields = ( 'password', 'password2', 'email', 'full_name')
  
    def validate(self, attrs):
        authentication_logs.info(f"Register User Request {attrs['email']}")
        if attrs['password'] != attrs['password2']:
            authentication_logs.error(f"Password fields didn't match. {attrs['email']}")
            raise serializers.ValidationError({"password": "Password fields didn't match."})

        if 'role' in attrs:
            roles = attrs['role'].split(',')
            attrs['role'] = [role.strip() for role in roles]
            authentication_logs.info(f"Role: {attrs['role']}")

        return attrs

    def create(self, validated_data):
        authentication_logs.info(f"Register User Request {validated_data['email']}")
        user = CustomUser.objects.create(
            email=validated_data['email']
        )
        user.set_password(validated_data['password'])
        roles = self.context.get('role', 'CADesigner')
        if isinstance(roles, str):
            roles = [roles]
        
        is_valid_role_added = False 
        for role in roles:
            if role in valid_roles:
                is_valid_role_added = True                
                
                group, created = Group.objects.get_or_create(name=role)
                user.groups.add(group)
        
        user.role = ", ".join(roles)
        user.full_name = validated_data['full_name']
        if not is_valid_role_added:
            role = 'CADesigner'
            group, created = Group.objects.get_or_create(name=role)
            user.groups.add(group)           
        
        user.save()
        authentication_logs.info(f"User Registered Successfully: {user.email}")
        return user

class ForgotPasswordSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)
    password2 = serializers.CharField(write_only=True)

    def validate(self, data):

        email = data.get("email")
        password = data.get("password")
        password2 = data.get("password2")


        if not CustomUser.objects.filter(email=email).exists():
            raise serializers.ValidationError({"email": "User with this email does not exist."})


        if password != password2:
            raise serializers.ValidationError({"password": "Passwords do not match."})

        return data

    def save(self):

        email = self.validated_data["email"]
        password = self.validated_data["password"]

        user = CustomUser.objects.get(email=email)
        user.set_password(password)  
        user.save()

class GetUserSerializer(serializers.ModelSerializer):
    role = serializers.SerializerMethodField()
    class Meta:
        model = CustomUser
        fields = ['id','email', 'role', 'full_name']
    
    def get_role(self, obj):        
        return [group.name for group in obj.groups.all()]

class UpdateUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomUser
        fields = ['full_name']