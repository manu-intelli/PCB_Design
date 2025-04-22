"""
URL configuration for pcb_design project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from rest_framework.permissions import IsAuthenticated
from django.contrib import admin
from django.urls import path,include
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from authentication.custom_authentication import CustomJWTAuthentication

schema_view = get_schema_view(
    openapi.Info(
        title="PCB Design API",
        default_version='v1',
        description="API documentation for the PCB Design project",
        terms_of_service="https://www.google.com/policies/terms/",
        contact=openapi.Contact(email="support@yourdomain.com"),
        license=openapi.License(name="MIT License"),
    ),
    public=True,
)

urlpatterns = [
    # create path for authentication app
    path('swagger/', schema_view.with_ui(), name='swagger-docs'),
    path('auth/', include('authentication.urls')),
    path('right-draw/', include('right_to_draw.urls')),
    path('masters/', include('masters.urls')),
    path('admin/', admin.site.urls),
]
