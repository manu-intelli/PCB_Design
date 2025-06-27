from django.urls import path
from .views import FilterUploadView,KPI_CalculationAPIView,SubmissionFilesListAPIView,SParameterPlotAPIView,StatisticalAndHistogramPlotAPIView

urlpatterns = [
    path('upload/', FilterUploadView.as_view(), name='filter-upload'),
    path('calculate-kpi/', KPI_CalculationAPIView.as_view(), name='kpi-calculation'),
    path('api/generate-s-parameter-plots/', SParameterPlotAPIView.as_view(), name='generate-s-parameter-plots'),
    path('api/generate-statistical-histogram-plots/', StatisticalAndHistogramPlotAPIView.as_view(), name='generate-statistical-histogram-plots'),
    path('api/submission-files-list/', SubmissionFilesListAPIView.as_view(), name='submission-files-list'),
]

