from django.db import models
from authentication.models import CustomUser
from utility.get_current_user import get_current_user



class BaseModel(models.Model):
    created_by = models.ForeignKey(CustomUser, on_delete=models.SET_NULL,
                                   related_name='%(class)s_created_by',
                                   null=True, blank=True, db_column='CREATED_BY')
    created_at = models.DateTimeField(auto_now_add=True, db_column='CREATED_AT')
    updated_by = models.ForeignKey(CustomUser, on_delete=models.SET_NULL,
                                   related_name='%(class)s_updated_by',
                                   null=True, blank=True, db_column='UPDATED_BY')
    updated_at = models.DateTimeField(auto_now=True, db_column='UPDATED_AT')

    class Meta:
        abstract = True

    def save(self, *args, **kwargs):
        user = get_current_user()
        if user and not self.pk:  # Set created_by for new records
            self.created_by = user
        if user:  # Set updated_by for both new and updated records
            self.updated_by = user
        super().save(*args, **kwargs)