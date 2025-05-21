from django.contrib import admin
from .models import CustomUser, Role
from django.contrib.auth.models import Permission
from django.contrib.auth.admin import UserAdmin
from django.utils.translation import gettext_lazy as _


class CustomUserAdmin(UserAdmin):
    model = CustomUser
    list_display = ("email", "full_name", "role", "get_roles_display", "is_staff", "is_active")  # changed
    list_filter = ("role", "is_staff", "is_active")

    fieldsets = (
        (_("User Info"), {"fields": ("email", "password")}),
        (_("Important Dates"), {"fields": ("last_login",)}),
        (_("Permissions"), {"fields": ("is_superuser", "groups", "user_permissions")}),
        (_("Personal Info"), {"fields": ("first_name", "last_name")}),
        (_("Status"), {"fields": ("is_staff", "is_active")}),
        (_("Date Joined"), {"fields": ("date_joined",)}),
        (_("Additional Info"), {"fields": ("is_logged_out", "role", "roles", "full_name")}),  # ⬅️ Added "roles"
    )

    add_fieldsets = (
        (None, {
            "classes": ("wide",),
            "fields": ("email", "full_name", "role", "roles", "password1", "password2", "is_staff", "is_active"),  # ⬅️ Added "roles"
        }),
    )

    search_fields = ("email", "full_name")
    ordering = ("email",)

    def get_roles_display(self, obj):
        return ", ".join([role.name for role in obj.roles.all()])

    get_roles_display.short_description = "Roles"  # Column name in admin list view

    def save_model(self, request, obj, form, change):
        if "password" in form.cleaned_data:
            obj.set_password(form.cleaned_data["password"])
        obj.save()

        
admin.site.register(CustomUser, CustomUserAdmin)
admin.site.register(Permission)
admin.site.register(Role)