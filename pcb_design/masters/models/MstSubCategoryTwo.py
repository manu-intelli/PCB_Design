from django.db import models

from .MstSubCategory import MstSubCategory
from .BaseModel import BaseModel
from authentication.alias import AliasField


class MstSubCategoryTwo(BaseModel):
    id = models.AutoField(primary_key=True, editable=False, db_column='ID')
    sub_2_category_name = models.CharField(max_length=255, db_column='SUB_2_CATEGORY_NAME')
    sub_category_id = models.ForeignKey(MstSubCategory, on_delete=models.CASCADE,
                                        related_name='subcategories2', db_column='SUB_CATEGORY_ID')
    type = models.CharField(max_length=25, blank=True, null=True, db_column='TYPE')
    name = AliasField(db_column='SUB_2_CATEGORY_NAME', blank=True, null=True,editable=False)

    class Meta:
        unique_together = ('sub_2_category_name', 'sub_category_id')
        verbose_name = '04 Sub Category Two'
        verbose_name_plural = '04 Sub Category Twos'

    def __str__(self):
        return self.sub_2_category_name
