from django.db import models

class FilterSubmission(models.Model):
    folder_name = models.CharField(max_length=255, unique=True)
    model_number = models.CharField(max_length=100)
    edu_number = models.CharField(max_length=100)
    filter_type = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.folder_name
