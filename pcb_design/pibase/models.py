from django.db import models
from django.core.exceptions import ValidationError
from django.conf import settings


class PiBaseFieldCategory(models.Model):
    """
    Represents a category for base fields, defining their input type.
    Examples include 'Model Family', 'Technology', etc., each associated with
    an input type like 'Dropdown' or 'Radio Button'.
    """
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
        """Returns the name of the field category."""
        return self.name

    class Meta:
        db_table = 'PiBaseFieldCategory'


class PiBaseFieldOption(models.Model):
    """
    Represents an option within a PiBaseFieldCategory.
    For example, for the 'Model Family' category, options might be 'iPhone', 'Galaxy', etc.
    """
    category = models.ForeignKey(PiBaseFieldCategory, on_delete=models.CASCADE, related_name='options')
    value = models.CharField(max_length=100)
    status = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    @property
    def count(self):
        """
        Returns the number of options within the same category.
        This property demonstrates how to provide additional computed data.
        """
        try:
            return PiBaseFieldOption.objects.filter(category=self.category).count()
        except Exception as e:
            # Log the exception for debugging purposes.
            # In a real application, you might use a proper logging framework.
            print(f"Error retrieving count for category {self.category.name}: {e}")
            # Depending on desired behavior, re-raise the exception or return a default/error value
            return 0 

    def __str__(self):
        """Returns a string representation of the option, including its category."""
        return f"{self.category.name} - {self.value}"

    class Meta:
        db_table = 'PiBaseFieldOption'


class PiBaseComponent(models.Model):
    """
    Represents a reusable component that can be associated with PiBaseRecords.
    Examples include specific ICs, connectors, etc.
    """
    name = models.CharField(max_length=100, unique=True)
    format = models.JSONField(default=dict, blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    icon = models.CharField(max_length=100, blank=True, null=True, help_text="Icon class for the component, e.g., 'CircuitBoard'")
    status = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        """Returns the name of the component."""
        return self.name

    class Meta:
        db_table = 'PiBaseComponent'


class PiBaseStatus(models.Model):
    """
    Defines various statuses for PiBaseRecords.
    """
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
        """Returns the human-readable status description."""
        return dict(self.STATUS_CHOICES).get(self.status_code, "Unknown")

    class Meta:
        db_table = 'PiBaseStatus'


class PiBaseImage(models.Model):
    """
    Stores image files related to PiBaseRecords, such as schematics or layouts.
    """
    image_type = models.CharField(max_length=50, help_text="e.g. schematic, layout")
    image_file = models.FileField(upload_to='schematics/', blank=True, null=True)
    cookies = models.CharField(max_length=100, blank=True, null=True)  # optional metadata

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        """Returns a descriptive string for the image."""
        return f"{self.image_type} Image"

    class Meta:
        db_table = 'PiBaseImage'


def get_default_status():
    """
    Helper function to retrieve the default status (e.g., 'Pending').
    This function handles the case where the default status might not exist yet.
    """
    try:
        return PiBaseStatus.objects.get(pk=1)
    except PiBaseStatus.DoesNotExist:
        # Log a warning if the default status is not found.
        # This might indicate a missing initial data setup.
        print("WARNING: Default PiBaseStatus (pk=1) does not exist. Please ensure it is created.")
        return None
    except Exception as e:
        # Catch any other unexpected errors during default status retrieval.
        print(f"ERROR: An unexpected error occurred while fetching default status: {e}")
        return None


class PiBaseRecord(models.Model):
    """
    The core model representing a complete record for a design or product,
    combining various base fields, components, and hardware details.
    """
    op_number = models.CharField(max_length=20, verbose_name="OP Number")
    opu_number = models.CharField(max_length=20, verbose_name="OPU Number")
    edu_number = models.CharField(max_length=20, verbose_name="EDU Number")
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
    via_holes_requirement = models.ForeignKey(PiBaseFieldOption, on_delete=models.SET_NULL, null=True,blank=True, related_name='device_via_holes_requirement', verbose_name="Via Holes Requirement")
    signal_launch_type = models.ForeignKey(PiBaseFieldOption, on_delete=models.SET_NULL, null=True,blank=True, related_name='device_signal_launch_type', verbose_name="Signal Launch Type")
    cover_type = models.ForeignKey(PiBaseFieldOption, on_delete=models.SET_NULL, null=True,blank=True, related_name='device_cover_type', verbose_name="Cover Type")
    design_rule_violation = models.ForeignKey(PiBaseFieldOption, on_delete=models.SET_NULL, null=True,blank=True, related_name='device_design_rule_violation', verbose_name="Design Rule Violation")

    impedance_selection = models.JSONField(default=dict, verbose_name="Impedance Selection")
    interfaces_details = models.JSONField(default=dict, blank=True, null=True, verbose_name="Interfaces Details")
    case_style_data = models.JSONField(default=dict, verbose_name="Case Style Data")

    components = models.ManyToManyField(PiBaseComponent, blank=True, related_name='pi_base_records_selected_components', verbose_name="Components")

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

    status = models.ForeignKey(PiBaseStatus, on_delete=models.SET_NULL,default=get_default_status, null=True, blank=True, related_name="pi_base_records_status", verbose_name="Status")
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True, related_name='pi_base_records')
    updated_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True, related_name='pi_base_records_updated')

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def clean(self):
        """
        Custom validation to ensure that related PiBaseFieldOption instances
        belong to the correct PiBaseFieldCategory.
        Raises ValidationError if an option is selected from the wrong category.
        """
        category_map = {
            'model_family': 'Model Family',
            'bottom_solder_mask': 'Bottom Solder Mask',
            'half_moon_requirement': 'Half Moon Requirement',
            'via_holes_requirement': 'Via Holes Requirement',
            'signal_launch_type': 'Signal Launch Type',
            'cover_type': 'Cover Type',
            'design_rule_violation': 'Design Rule Violation',
        }
        errors = {}
        for field_name, expected_category in category_map.items():
            option = getattr(self, field_name)
            if option:
                try:
                    if option.category.name != expected_category:
                        errors[field_name] = f"Option must belong to category '{expected_category}'."
                except PiBaseFieldCategory.DoesNotExist:
                    # This implies a data inconsistency where an option's category is missing.
                    # This should be a rare case if referential integrity is maintained.
                    errors[field_name] = f"Associated category for '{field_name}' (ID: {option.category_id}) not found."
                except Exception as e:
                    # Catch any other unexpected errors during category validation (e.g., database issues).
                    errors[field_name] = f"An unexpected error occurred validating {field_name}: {e}"

        if errors:
            raise ValidationError(errors)

    def __str__(self):
        """Returns a human-readable string representation of the PiBaseRecord."""
        return f"{self.model_name} (OP#{self.op_number})"

    class Meta:
        verbose_name = "PiBaseRecord"
        verbose_name_plural = "PiBaseRecords"
        ordering = ['-created_at']
        db_table = 'PiBaseRecord'