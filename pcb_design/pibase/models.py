from django.db import models
from django.core.exceptions import ValidationError
from django.conf import settings


class PiBaseFieldCategory(models.Model):
    INPUT_TYPE_CHOICES = (
        (1, 'Dropdown'),
        (2, 'Radio Button'),
    )

    name = models.CharField(max_length=100, unique=True)
    input_type = models.PositiveSmallIntegerField(choices=INPUT_TYPE_CHOICES)
    status = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    class Meta:
        db_table = 'PiBaseFieldCategory'


class PiBaseFieldOption(models.Model):
    category = models.ForeignKey(PiBaseFieldCategory, on_delete=models.CASCADE, related_name='options')
    value = models.CharField(max_length=100)
    status = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    @property
    def count(self):
        return PiBaseFieldOption.objects.filter(category=self.category).count()

    def __str__(self):
        return f"{self.category.name} - {self.value}"

    class Meta:
        db_table = 'PiBaseFieldOption'


class PiBaseComponent(models.Model):
    name = models.CharField(max_length=100, unique=True)
    format = models.JSONField(default=dict, blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    icon = models.CharField(max_length=100, blank=True, null=True, help_text="Icon class for the component, e.g., 'CircuitBoard'")
    status = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    class Meta:
        db_table = 'PiBaseComponent'


class PiBaseStatus(models.Model):
    STATUS_CHOICES = (
        (1, 'Pending'),
        (2, 'Complete'),
        (3, 'Close'),
        (4, 'Delete'),
    )

    status_code = models.PositiveSmallIntegerField(choices=STATUS_CHOICES, unique=True)
    description = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return dict(self.STATUS_CHOICES).get(self.status_code, "Unknown")

    class Meta:
        db_table = 'PiBaseStatus'



class PiBaseImage(models.Model):
    image_type = models.CharField(max_length=50, help_text="e.g. schematic, layout")
    image_file = models.FileField(upload_to='schematics/', blank=True, null=True)
    cookies = models.CharField(max_length=100, blank=True, null=True)  # optional metadata

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.image_type} Image"

    class Meta:
        db_table = 'PiBaseImage'

class PiBaseRecord(models.Model):
    op_no = models.CharField(max_length=20, verbose_name="OP Number")
    opu_no = models.CharField(max_length=20, verbose_name="OPU Number")
    edu_no = models.CharField(max_length=20, verbose_name="EDU Number")
    model_name = models.CharField(max_length=100)
    schematic = models.ForeignKey(
        PiBaseImage, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True, 
        related_name='schematic_records',
        help_text="Schematic image reference"
    )
    similar_model_layout = models.TextField(blank=True, null=True)
    revision_number = models.CharField(max_length=20)

    technology = models.ForeignKey(PiBaseFieldOption, on_delete=models.SET_NULL, null=True,blank=True, related_name='device_technology', verbose_name="Technology")
    model_family = models.ForeignKey(PiBaseFieldOption, on_delete=models.SET_NULL, null=True,blank=True, related_name='device_model_family', verbose_name="Model Family")
    bottom_solder_mask = models.ForeignKey(PiBaseFieldOption, on_delete=models.SET_NULL, null=True,blank=True ,related_name='device_bottom_solder_mask', verbose_name="Bottom Solder Mask")
    half_moon_requirement = models.ForeignKey(PiBaseFieldOption, on_delete=models.SET_NULL, null=True,blank=True, related_name='device_half_moon_requirement', verbose_name="Half Moon Requirement")
    via_holes_on_signal_pads = models.ForeignKey(PiBaseFieldOption, on_delete=models.SET_NULL, null=True,blank=True, related_name='device_via_holes_on_signal_pads', verbose_name="Via Holes Requirement")
    signal_launch_type = models.ForeignKey(PiBaseFieldOption, on_delete=models.SET_NULL, null=True,blank=True, related_name='device_signal_launch_type', verbose_name="Signal Launch Type")
    cover_type = models.ForeignKey(PiBaseFieldOption, on_delete=models.SET_NULL, null=True,blank=True, related_name='device_cover_type', verbose_name="Cover Type")
    design_rule_violation_accepted = models.ForeignKey(PiBaseFieldOption, on_delete=models.SET_NULL, null=True,blank=True, related_name='device_design_rule_violation_accepted', verbose_name="Design Rule Violation")

    impedance_selection = models.JSONField(default=dict, verbose_name="Impedance Selection")
    interfaces_details = models.JSONField(default=dict, blank=True, null=True, verbose_name="Interfaces Details")
    case_style_data = models.JSONField(default=dict, verbose_name="Case Style Data")

    components = models.ManyToManyField(PiBaseComponent, blank=True, related_name='devices')

    # Hardware fields
    can_details = models.JSONField(default=dict, blank=True, null=True)
    pcb_details = models.JSONField(default=dict, blank=True, null=True)
    chip_aircoil_details = models.JSONField(default=dict, blank=True, null=True)
    chip_inductor_details = models.JSONField(default=dict, blank=True, null=True)
    chip_capacitor_details = models.JSONField(default=dict, blank=True, null=True)
    chip_resistor_details = models.JSONField(default=dict, blank=True, null=True)
    transformer_details = models.JSONField(default=dict, blank=True, null=True)
    shield_details = models.JSONField(default=dict, blank=True, null=True)
    finger_details = models.JSONField(default=dict, blank=True, null=True)
    copper_flaps_details = models.JSONField(default=dict, blank=True, null=True)
    resonator_details = models.JSONField(default=dict, blank=True, null=True)
    ltcc_details = models.JSONField(default=dict, blank=True, null=True)

    special_requirements = models.TextField(blank=True, null=True, verbose_name="Special Requirements")

    status = models.ForeignKey(PiBaseStatus, on_delete=models.SET_NULL,default=1, null=True, blank=True, related_name="device_models")
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True, related_name='pi_base_records')
    updated_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True, related_name='pi_base_records_updated')

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def clean(self):
        category_map = {
            'model_family': 'Model Family',
            'bottom_solder_mask': 'Bottom Solder Mask',
            'half_moon_requirement': 'Half Moon Requirement',
            'via_holes_on_signal_pads': 'Via Holes On Signal Pads',
            'signal_launch_type': 'Signal Launch Type',
            'cover_type': 'Cover Type',
            'design_rule_violation_accepted': 'Design Rule Violation Accepted',
        }
        for field_name, expected_category in category_map.items():
            option = getattr(self, field_name)
            if option and option.category.name != expected_category:
                raise ValidationError({
                    field_name: f"Option must belong to category '{expected_category}'."
                })

    def __str__(self):
        return f"{self.model_name} (OP#{self.op_no})"

    class Meta:
        verbose_name = "PiBaseRecord"
        verbose_name_plural = "PiBaseRecords"
        ordering = ['-created_at']
        db_table = 'PiBaseRecord'
