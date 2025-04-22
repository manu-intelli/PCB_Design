from django.contrib import admin
from .models import CADDesignTemplates, CADVerifierTemplates, CADApproverTemplates

admin.site.register(CADDesignTemplates)
admin.site.register(CADVerifierTemplates)
admin.site.register(CADApproverTemplates)
