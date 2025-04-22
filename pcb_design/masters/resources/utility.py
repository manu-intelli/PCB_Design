from import_export.widgets import ForeignKeyWidget
from django.utils import timezone
from import_export.widgets import ForeignKeyWidget, ManyToManyWidget
from django.core.exceptions import ObjectDoesNotExist
from django.db import models


class CustomForeignKeyWidget(ForeignKeyWidget):
    def __init__(self, model, field='id'):        
        super().__init__(model, field)
    
    def clean(self, value, row=None, *args, **kwargs):
        if value:            
            try:
                response =  super().clean(value, row, *args, **kwargs)
                return response
            except ObjectDoesNotExist as e:                
                try:
                    value = int(value)
                    res = self.model.objects.get(**{'id': value})               
                    return res
                except ObjectDoesNotExist:
                    raise ValueError(f"{self.model.__name__} with {self.field}={value} does not exist.")
        return None


class CustomManyToManyWidget(ManyToManyWidget):
    def __init__(self, model, separator=None, field=None, **kwargs):        
        if separator is None:
            separator = ","
        if field is None:
            field = "id"
        self.model = model
        self.separator = separator
        self.field = field
        super().__init__(self.model, self.separator, self.field, **kwargs)

    def clean(self, value, row=None, **kwargs):        
        if not value:
            return self.model.objects.none()        
        if isinstance(value, (float, int)):
            ids = [int(value)]
        else:
            ids = value.split(self.separator)
            # This Needs to change If you change Primary Key from AutoField to Other Field
            ids = filter(None, [int(i.strip()) for i in ids])            
        res = self.model.objects.filter(**{"%s__in" % self.field: ids})        
        return res


def before_save_instance_update_create_date(instance, model):
    try:
        existing_instance = model.objects.get(pk=instance.pk)
        instance.created_at = existing_instance.created_at if \
            existing_instance.created_at else timezone.now()
    except Exception as ex:
        instance.created_at = timezone.now()

    # Strip spaces for all CharField and TextField fields
    for field in instance._meta.fields:
        if isinstance(field, (models.CharField, models.TextField)):
            if field is None:
                continue
            value = getattr(instance, field.name)
            if isinstance(value, str):
                # Remove leading and trailing spaces from string fields
                setattr(instance, field.name, value.strip())

    return instance
