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
    role = serializers.ListField(
        child=serializers.CharField(), required=False  # Accept role names
    )

    class Meta:
        model = CustomUser
        fields = ('email', 'full_name', 'password', 'password2', 'role')

    def validate(self, attrs):
        if attrs['password'] != attrs['password2']:
            raise serializers.ValidationError({"password": "Password fields didn't match."})
        return attrs

    def create(self, validated_data):
        roles = validated_data.pop('role', ['CADesigner'])
        password = validated_data.pop('password')
        validated_data.pop('password2')

        cleaned_roles = [r.strip() for r in roles if r.strip()]
        if len(cleaned_roles) > 1 and 'NormalUser' in cleaned_roles:
            cleaned_roles.remove('NormalUser')

        group_ids = []
        user = CustomUser.objects.create(**validated_data)
        user.set_password(password)

        for role in cleaned_roles:
            try:
                group = Group.objects.get(name=role)
                user.groups.add(group)
                group_ids.append(str(group.id))  # collect the group ID
            except Group.DoesNotExist:
                continue  # skip invalid roles

        # Fallback if no valid roles
        if not group_ids:
            group, _ = Group.objects.get_or_create(name='NormalUser')
            user.groups.add(group)
            group_ids.append(str(group.id))

        # Save group IDs as comma-separated string in the `role` CharField
        user.role = ",".join(group_ids)
        user.save()

        return user

    def to_representation(self, instance):
        rep = super().to_representation(instance)
        # Send role names in response
        rep['role'] = [group.name for group in instance.groups.all()]
        return rep



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


class RoleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Group
        fields = ['id', 'name']  # adapt fields as per your Role model