from django.db import models

from .BaseModel import BaseModel
from .MstDesignOptions import MstDesignOptions
from .MstSectionRules import MstSectionRules
from authentication.alias import AliasField


class MstSectionGroupings(BaseModel):
    id = models.AutoField(primary_key=True, editable=False, db_column='ID')
    design_doc = models.CharField(max_length=255, db_column='DESIGN_DOC')
    section_name = models.CharField(max_length=255, unique=True, db_column='SECTION_NAME')
    rules = models.ManyToManyField(MstSectionRules, db_column='RULES', blank=True, null=True)
    design_options = models.ManyToManyField(MstDesignOptions, db_column='DESIGN_OPTIONS', related_name='section_groups', 
                                            blank=True, null=True)
    name = AliasField(db_column='SECTION_NAME', blank=True, null=True, editable=False)

    class Meta:
        verbose_name = '07 Section Group'
        verbose_name_plural = '07 Section Groups'
        unique_together = ('design_doc', 'section_name')

    def __str__(self):
        return f"{self.section_name}-{self.design_doc}"

