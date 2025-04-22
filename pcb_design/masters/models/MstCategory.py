from django.db import models
from .MstComponet import MstComponent
from .BaseModel import BaseModel
from authentication.alias import AliasField

class MstCategory(BaseModel):
    id = models.AutoField(primary_key=True, editable=False, db_column='ID')
    category_name = models.CharField(max_length=255, db_column='CATEGORY_NAME')
    component_Id = models.ForeignKey(MstComponent, on_delete=models.CASCADE,
                                     related_name='component_categories', db_column='COMPONENT_ID')
    is_verifier = models.BooleanField(default=True, db_column='IS_VERIFIER')
    name = AliasField(db_column='CATEGORY_NAME', blank=True, null=True, editable=False)

    
    class Meta:
        unique_together = ('category_name', 'component_Id')  
        verbose_name = '02 Category'
        verbose_name_plural = '02 Categories'
        
    def __str__(self):
        return self.category_name

