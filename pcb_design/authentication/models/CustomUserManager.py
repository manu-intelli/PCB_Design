from django.contrib.auth.models import BaseUserManager
from django.contrib.auth.models import Group


class CustomUserManager(BaseUserManager):
    def create_user(self, email, password=None,**extra_fields):
        if not email:
            raise ValueError("The Email field must be set")
        user = self.model(email=self.normalize_email(email), **extra_fields)
        user.set_password(password)
        user.save()
        return user

    def create_superuser(self, email, password,**extra_fields):
        
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)
        extra_fields.setdefault("is_active", True)

        if extra_fields.get("is_staff") is not True:
            raise ValueError(_("Superuser must have is_staff=True."))
        if extra_fields.get("is_superuser") is not True:
            raise ValueError(_("Superuser must have is_superuser=True."))
        
        superuser_group, created = Group.objects.get_or_create(name='Admin')        
        user = self.create_user(email, password, **extra_fields)
        user.groups.add(superuser_group)
        user.is_staff = True
        user.role = 'Admin'
        user.save()

        return user
