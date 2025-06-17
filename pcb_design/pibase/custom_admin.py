"""
Custom admin views for the pibase app.

This module provides a custom admin page to view and delete unused PiBaseImage records.
Includes error handling and clear documentation.

Views:
    - my_custom_admin_view: Shows unused images and allows bulk deletion.
    - MyAdminSite: Custom admin site with the custom page registered.
"""

from django.contrib.admin.views.decorators import staff_member_required
from django.shortcuts import render, redirect
from django.urls import path
from django.contrib import admin

from pibase.models import PiBaseImage, PiBaseRecord
from django.template.response import TemplateResponse


@staff_member_required
def my_custom_admin_view(request):
    """
    Custom admin view to display and delete unused PiBaseImage records.

    GET: Show all unused images.
    POST (action=delete_all): Delete all unused images and their files.
    """
    try:
        # Get all used schematic image IDs
        used_image_ids = PiBaseRecord.objects.values_list('schematic_id', flat=True)

        # Get unused images
        unused_images = PiBaseImage.objects.exclude(id__in=used_image_ids)

        # Handle deletion
        if request.method == "POST" and request.POST.get("action") == "delete_all":
            count = unused_images.count()

            # Delete associated files from media folder
            for image in unused_images:
                try:
                    if image.image_file:
                        image.image_file.delete(save=False)  # Deletes the file, not the DB entry
                except Exception as file_err:
                    # Continue deleting other files even if one fails
                    pass

            # Now delete the database entries
            try:
                unused_images.delete()
            except Exception as db_err:
                return render(request, 'admin/custom_page.html', {
                    "unused_images": unused_images,
                    "deleted_count": 0,
                    "error": f"Error deleting records: {db_err}"
                })

            return redirect(request.path + f"?deleted={count}")

        deleted_count = request.GET.get("deleted")

        return render(request, 'admin/custom_page.html', {
            "unused_images": unused_images,
            "deleted_count": deleted_count,
        })
    except Exception as e:
        return render(request, 'admin/custom_page.html', {
            "unused_images": [],
            "deleted_count": 0,
            "error": f"Unexpected error: {e}"
        })


class MyAdminSite(admin.AdminSite):
    """
    Custom admin site with a custom page for managing unused images.
    """
    site_header = "Custom Admin Panel"
    site_title = "Admin"
    index_title = "Welcome to Custom Admin"

    def get_urls(self):
        """
        Register custom admin URLs.
        """
        urls = super().get_urls()
        custom_urls = [
            path('', self.admin_view(my_custom_admin_view), name='custom-page'),
        ]
        return custom_urls + urls


# Create an instance of your custom admin site
admin_site = MyAdminSite(name='myadmin')
