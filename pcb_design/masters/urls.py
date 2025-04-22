from django.urls import path
from .views import Components

urlpatterns = [
    path('components/<int:component_id>/', Components.as_view(), name='component-detail'), # Fetch a single component by id
    path('components/', Components.as_view(), name='component-list'), # Fetch all components
]
