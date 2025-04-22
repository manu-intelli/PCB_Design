from django.contrib.auth.models import AbstractUser
from django.db import models
from .CustomUserManager import CustomUserManager


VALID_ROLES = [
        ('Admin', 'Admin'),
        ('CADesigner', 'CADesigner'),
        ('Approver', 'Approver'),
        ('Verifier', 'Verifier'),
    ]

class CustomUser(AbstractUser):  
    username = None
    email = models.EmailField(unique=True)
    is_logged_out = models.BooleanField(default=True)
    role = models.CharField(max_length=20, choices=VALID_ROLES, default='CADesigner', blank=True, null=True)
    full_name = models.CharField(max_length=255)
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['full_name']

    objects = CustomUserManager()

    def __str__(self):
        return self.email
