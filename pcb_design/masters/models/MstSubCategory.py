from django.db import models

from .MstCategory import MstCategory
from .BaseModel import BaseModel
from authentication.alias import AliasField


class MstSubCategory(BaseModel):
    id = models.AutoField(primary_key=True, editable=False, db_column='ID')
    sub_category_name = models.CharField(max_length=255, db_column='SUB_CATEGORY_NAME')
    category_Id = models.ForeignKey(MstCategory, on_delete=models.CASCADE, related_name='subcategories',
                                    db_column='CATEGORY_ID')
    is_verifier = models.BooleanField(default=True, db_column='IS_VERIFIER')
    name = AliasField(db_column='SUB_CATEGORY_NAME', blank=True, null=True, editable=False)

    class Meta:
        unique_together = ('sub_category_name', 'category_Id') 
        verbose_name = '03 Sub Category'
        verbose_name_plural = '03 Sub Categories' 

    def __str__(self):
        return f"{self.sub_category_name}"
