import os
from django.db import models
from django.conf import settings

class FilterSubmission(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, blank=True)
    folder_name = models.CharField(max_length=255, unique=True)
    model_number = models.CharField(max_length=100)
    edu_number = models.CharField(max_length=100)
    filter_type = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.folder_name

    @property
    def folder_path(self):
        return f"uploads/{self.folder_name}"

    def list_files_in_folder(self):
        folder_full_path = os.path.join(settings.MEDIA_ROOT, 'uploads', self.folder_name)
        if not os.path.exists(folder_full_path):
            return "Folder not found."

        file_list = []
        for root, dirs, files in os.walk(folder_full_path):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), settings.MEDIA_ROOT)
                file_list.append(f"/media/{rel_path}")

        if file_list:
            return '\n'.join(file_list)
        return "No files found."

    list_files_in_folder.short_description = "Files in Folder"
