from django.db import models

from .MstComponet import MstComponent
from .MstCategory import MstCategory
from .MstSubCategory import MstSubCategory
from .BaseModel import BaseModel
from authentication.alias import AliasField


class MstVerifierField(BaseModel):
    id = models.AutoField(primary_key=True, editable=False, db_column='ID')
    component = models.ForeignKey(MstComponent, on_delete=models.DO_NOTHING, db_column='COMPONENT_ID', blank=True, null=True)
    category = models.ForeignKey(MstCategory, on_delete=models.DO_NOTHING, db_column='CATEGORY_ID', blank=True, null=True)
    sub_category = models.ManyToManyField(MstSubCategory, db_column='SUB_CATEGORY_ID', blank=True, null=True)
    field_name = models.CharField(max_length=255, db_column='FIELD_NAME')
    name = AliasField(db_column='FIELD_NAME', blank=True, null=True, editable=False)
    
    class Meta:        
        verbose_name = '08 Verifier Field'
        verbose_name_plural = '08 Verifier Fields'
        unique_together = ('component', 'category', 'field_name')

    def __str__(self):
        return self.field_name
