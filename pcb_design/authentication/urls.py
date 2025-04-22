from django.urls import path
from .views import LoginView, LogoutView, UserRegistrationView,ForgetPasswordView,UserAPIView

urlpatterns = [
    path('register/', UserRegistrationView.as_view(), name='user-registration'),    
    path('login/', LoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),  
    path('forgot-password/',ForgetPasswordView.as_view(),name='forgot-password'),
    path('users/', UserAPIView.as_view(), name='get-users'),
    path('users/<int:pk>/', UserAPIView.as_view(), name='crud-users'),
    
]
