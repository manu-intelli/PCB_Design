from django.db import models
from .MstSubCategoryTwo import MstSubCategory
from .BaseModel import BaseModel
from authentication.alias import AliasField


class MstDesignOptions(BaseModel):
    id = models.AutoField(primary_key=True, editable=False, db_column='ID')
    desing_option_name = models.CharField(max_length=255, db_column='DESIGN_OPTION_NAME')
    sub_category_id = models.ForeignKey(MstSubCategory, on_delete=models.CASCADE,
                                     related_name='design_categories', db_column='SUB_CATEGORY_ID')
    name = AliasField(db_column='DESIGN_OPTION_NAME', blank=True, null=True, editable=False)

    
    class Meta:
        unique_together = ('desing_option_name', 'sub_category_id')
        verbose_name = '05 DesignOption'
        verbose_name_plural = '05 DesignOptions'
        
    def __str__(self):
        return self.desing_option_name
