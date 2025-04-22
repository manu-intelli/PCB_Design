import warnings
from django.db import models


class AliasField(models.Field):
    def contribute_to_class(self, cls, name, virtual_only=False):
        super(AliasField, self).contribute_to_class(cls, name, private_only=True)
        setattr(cls, name, self)

    def __get__(self, instance, instance_type=None):
        return getattr(instance, self.db_column)
