from django.urls import path
from .views import FilterUploadView,KPI_CalculationAPIView,PlotGenerationAPIView

urlpatterns = [
    path('upload/', FilterUploadView.as_view(), name='filter-upload'),
    path('calculate-kpi/', KPI_CalculationAPIView.as_view(), name='kpi-calculation'),
    path('api/generate-plots/', PlotGenerationAPIView.as_view(), name='generate-plots'),
]
