from django.contrib.admin.views.decorators import staff_member_required
from django.shortcuts import render, redirect
from django.urls import path
from django.contrib import admin

from pibase.models import PiBaseImage, PiBaseRecord  # ✅ Import models
from django.template.response import TemplateResponse


@staff_member_required
def my_custom_admin_view(request):
    # Get all used schematic image IDs
    used_image_ids = PiBaseRecord.objects.values_list('schematic_id', flat=True)

    # Get unused images
    unused_images = PiBaseImage.objects.exclude(id__in=used_image_ids)

    # Handle deletion
    if request.method == "POST" and request.POST.get("action") == "delete_all":
        count = unused_images.count()

        # Delete associated files from media folder
        for image in unused_images:
            if image.image_file:
                image.image_file.delete(save=False)  # Deletes the file, not the DB entry

        # Now delete the database entries
        unused_images.delete()

        return redirect(request.path + f"?deleted={count}")

    deleted_count = request.GET.get("deleted")

    return render(request, 'admin/custom_page.html', {
        "unused_images": unused_images,
        "deleted_count": deleted_count,
    })

class MyAdminSite(admin.AdminSite):
    site_header = "Custom Admin Panel"
    site_title = "Admin"
    index_title = "Welcome to Custom Admin"

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('', self.admin_view(my_custom_admin_view), name='custom-page'),
        ]
        return custom_urls + urls


# Create an instance of your custom admin site
admin_site = MyAdminSite(name='myadmin')
