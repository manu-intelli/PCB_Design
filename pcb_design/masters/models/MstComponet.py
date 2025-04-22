from django.db import models
from .BaseModel import BaseModel
from authentication.alias import AliasField


class MstComponent(BaseModel):
    id = models.AutoField(primary_key=True, editable=False, db_column='ID')
    component_name = models.CharField(max_length=255, unique=True, db_column='COMPONENT_NAME')
    description = models.CharField(max_length=255, null=True, blank=True, db_column='DESCRIPTION')
    name = AliasField(db_column='COMPONENT_NAME', blank=True, null=True, editable=False)

    class Meta:
        verbose_name = '01 Component'
        verbose_name_plural = '01 Components'
    
    def __str__(self):
        return self.component_name
