from django.contrib import admin
from .models import CustomUser
from django.contrib.auth.models import Permission
from django.contrib.auth.admin import UserAdmin
from django.utils.translation import gettext_lazy as _


class CustomUserAdmin(UserAdmin):
    model = CustomUser
    list_display = ("email", "full_name", "role", "is_staff", "is_active")
    list_filter = ("role", "is_staff", "is_active")

    fieldsets = (
        (_("User Info"), {"fields": ("email", "password")}),  # Email & Password
        (_("Important Dates"), {"fields": ("last_login",)}),  # Last Login
        (_("Permissions"), {"fields": ("is_superuser", "groups", "user_permissions")}),  # Superuser, Groups & Permissions
        (_("Personal Info"), {"fields": ("first_name", "last_name")}),  # First Name & Last Name
        (_("Status"), {"fields": ("is_staff", "is_active")}),  # Staff & Active Status
        (_("Date Joined"), {"fields": ("date_joined",)}),  # Date Joined
        (_("Additional Info"), {"fields": ("is_logged_out", "role", "full_name")}),  # Logged Out, Role & Full Name
    )

    add_fieldsets = (
        (None, {
            "classes": ("wide",),
            "fields": ("email", "full_name", "role", "password1", "password2", "is_staff", "is_active"),
        }),
    )

    search_fields = ("email", "full_name")
    ordering = ("email",)

    def save_model(self, request, obj, form, change):
        """Ensure password is hashed before saving"""
        if "password" in form.cleaned_data:
            obj.set_password(form.cleaned_data["password"])  # Hash password before saving
        obj.save()
        
admin.site.register(CustomUser, CustomUserAdmin)
admin.site.register(Permission)