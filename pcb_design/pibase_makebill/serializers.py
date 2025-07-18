import uuid
from rest_framework import serializers
from pibase.models import PiBaseRecord, PiBaseComponent
from .models import MakeBillRecord, MakeBillStatus
from pibase.serializers import PiBaseRecordSerializer
from collections import defaultdict

class PiBaseToMakeBillRecordSerializer(serializers.ModelSerializer):
    """
    Serializer for PiBaseRecord model with UUID-based recordId and user's full name.
    """

    recordId = serializers.SerializerMethodField()
    created_by_name = serializers.SerializerMethodField()

    class Meta:
        model = PiBaseRecord
        fields = [
            *[f.name for f in PiBaseRecord._meta.fields],
            "recordId",
            "created_by_name",
        ]

    def get_recordId(self, obj):
        """
        Generates a UUID (version 5) for the recordId using the DNS namespace and the PiBaseRecord's ID.
        
        Args:
            obj (PiBaseRecord): The PiBaseRecord instance.

        Returns:
            str: A UUID string representing the recordId.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"PiBase-{obj.id}"))

    def get_created_by_name(self, obj):
        """
        Retrieves the full name of the user who created the PiBaseRecord.
        If the full name is not available, it falls back to the username, then the string representation of the user.

        Args:
            obj (PiBaseRecord): The PiBaseRecord instance.

        Returns:
            str: The full name, username, or string representation of the creating user, or an empty string if no user is associated.
        """
        user = getattr(obj, "created_by", None)
        if user:
            full_name = user.get_full_name()
            return full_name or user.username or str(user)
        return ""


class PiBaseComponentSerializer(serializers.ModelSerializer):
    """
    Serializer for the PiBaseComponent model, exposing all its fields.
    """
    class Meta:
        model = PiBaseComponent
        fields = "__all__"


COMPONENT_TO_FIELD_MAP = {
    "PCB Name": "pcb_details",
    "CAN Details": "can_details",
    "Chip Capacitors": "chip_capacitor_details",
    "Chip Inductors": "chip_inductor_details",
    "Air Coil": "chip_aircoil_details",
    "Transformer": "transformer_details",
    "Shield": "shield_details",
    "Finger Details": "finger_details",
    "Copper Flap Details": "copper_flaps_details",
    "Resonator Details": "resonator_details",
    "Chip Resistor": "chip_resistor_details",
    "LTCC Details": "ltcc_details",
}



class MakeBillRecordGetSerializer(serializers.Serializer):
    """
    Serializer for retrieving MakeBillRecord details, including associated PiBaseRecord
    information and a structured list of components with their details.
    """
    pibaseId = serializers.SerializerMethodField()
    componentsData = serializers.SerializerMethodField()
    pibaseRecord = serializers.SerializerMethodField()

    def get_pibaseId(self, obj):
        """
        Returns the ID of the PiBaseRecord instance.

        Args:
            obj (PiBaseRecord): The PiBaseRecord instance.

        Returns:
            int: The ID of the PiBaseRecord.
        """
        return obj.id

    def get_componentsData(self, obj):
        """
        Processes and structures component data from the PiBaseRecord, generating part numbers
        and descriptions based on component type and associated details. It also handles
        the generation of sequential counters for new (missing part number) components.

        Args:
            obj (PiBaseRecord): The PiBaseRecord instance.

        Returns:
            list: A list of dictionaries, each representing a component with detailed information
                  for the Make Bill.
        """
        components = obj.components.all() if hasattr(obj.components, "all") else obj.components or []
        selected_fields = [
            COMPONENT_TO_FIELD_MAP.get(comp.name) for comp in components
            if COMPONENT_TO_FIELD_MAP.get(comp.name)
        ]

        combined_data = []
        s_no = 1
        case_style_data = obj.case_style_data or {}
        edu_number = obj.edu_number
        opu_number = obj.opu_number

        # Count only missing part numbers for counter
        component_counters = defaultdict(int)

        for comp in components:
            field = COMPONENT_TO_FIELD_MAP.get(comp.name)
            if not field:
                continue

            details = getattr(obj, field, None)
            if not details:
                continue

            flat_items = []

            if isinstance(details, list):
                flat_items.extend(details)
            elif isinstance(details, dict):
                for v in details.values():
                    if isinstance(v, list):
                        flat_items.extend(v)
                    elif isinstance(v, dict):
                        flat_items.append(v)
                if not any(isinstance(v, (list, dict)) for v in details.values()):
                    flat_items.append(details)

            for item in flat_items:
                if not item.get("bPartNumber", "").strip():
                    component_counters[comp.name] += 1

        # Build final rows
        current_counters = defaultdict(int)
        for comp in components:
            field = COMPONENT_TO_FIELD_MAP.get(comp.name)
            if not field:
                continue

            details = getattr(obj, field, None)
            if not details:
                continue

            flat_items = []

            if isinstance(details, list):
                flat_items.extend(details)
            elif isinstance(details, dict):
                for v in details.values():
                    if isinstance(v, list):
                        flat_items.extend(v)
                    elif isinstance(v, dict):
                        flat_items.append(v)
                if not any(isinstance(v, (list, dict)) for v in details.values()):
                    flat_items.append(details)

            for item in flat_items:
                # Only increment and assign counter if part number is missing
                counter_str = ""
                b_part_number = item.get("bPartNumber", "").strip().upper()
                part_type = item.get("partType", "").strip()

                if b_part_number in ["", "TBD"] or part_type == "New":
                    current_counters[comp.name] += 1
                    counter_str = str(current_counters[comp.name]).zfill(2)
                combined_data.append(
                    _make_component_row(item, s_no, comp.name, case_style_data, edu_number, opu_number, counter_str)
                )
                s_no += 1

        return combined_data

    def get_pibaseRecord(self, obj):
        """
        Serializes the entire PiBaseRecord object using the PiBaseRecordSerializer.

        Args:
            obj (PiBaseRecord): The PiBaseRecord instance.

        Returns:
            dict: The serialized data of the PiBaseRecord.
        """
        return PiBaseRecordSerializer(obj).data

    def to_representation(self, instance):
        """
        Overrides the default `to_representation` method to include custom fields
        and append 'Case Style' and 'Special Requirements' data if available.

        Args:
            instance (PiBaseRecord): The PiBaseRecord instance to be serialized.

        Returns:
            dict: The complete serialized representation of the MakeBillRecord,
                  including PiBaseRecord details and structured component data.
        """
        components_data = self.get_componentsData(instance)

        if instance.case_style_data:
            edu_code = instance.edu_number or "XXXX"
            components_data.append({
                "id": str(len(components_data) + 1),
                "sNo": len(components_data) + 1,
                "component": "Case Style",
                "componentName": "Case Style",
                "partNo": f"99-01-EDU{edu_code}",
                "rev": "------",
                "partDescription": "-------",
                "qtyPerUnit": "0",
                "stdPkgQty": "0",
                "qtyRequired": "0",
                "issuedQty": "",
                "qNo": "",
                "comments": "Leave your comment",
                "notes": "",
            })

        if instance.special_requirements:
            sr_text = instance.special_requirements if isinstance(instance.special_requirements, str) else str(instance.special_requirements)
            components_data.append({
                "id": str(len(components_data) + 1),
                "sNo": len(components_data) + 1,
                "component": "Special Requirements",
                "componentName": "Special Requirements",
                "partNo": "------",
                "rev": "------",
                "partDescription": sr_text,
                "qtyPerUnit": "0",
                "stdPkgQty": "0",
                "qtyRequired": "0",
                "issuedQty": "",
                "qNo": "",
                "comments": "Leave your comment",
                "notes": "",
            })

        return {
            "pibaseId": self.get_pibaseId(instance),
            "recordId": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"PiBase-{instance.id}")),
            "model_name": instance.model_name,
            "op_number": instance.op_number,
            "opu_number": instance.opu_number,
            "edu_number": instance.edu_number,
            "revision_number": 1,
            "created_by_name": instance.created_by.get_full_name() if instance.created_by else "",
            "created_at": instance.created_at.isoformat() if instance.created_at else None,
            "updated_at": instance.updated_at.isoformat() if instance.updated_at else None,
            "componentsData": components_data,
            "pibaseRecord": self.get_pibaseRecord(instance)
        }


def _make_component_row(item, s_no, component_name=None, case_style_data=None, edu_number=None, opu_number=None, counter_str=""):
    """
    Helper function to construct a single row for a component in the Make Bill.
    It generates part numbers and descriptions based on the component type and
    whether a part number is already present.

    Args:
        item (dict): The dictionary containing details of the individual component.
        s_no (int): The serial number for the component row.
        component_name (str, optional): The name of the component type (e.g., "PCB Name"). Defaults to None.
        case_style_data (dict, optional): Dictionary containing case style dimensions. Defaults to None.
        edu_number (str, optional): The EDU number associated with the record. Defaults to None.
        opu_number (str, optional): The OPU number associated with the record. Defaults to None.
        counter_str (str, optional): A zero-padded string counter for new parts. Defaults to "".

    Returns:
        dict: A dictionary representing a single row of component data for the Make Bill.
    """
    part_no = item.get("bPartNumber", "").strip()
    part_description = ""
    comments = item.get("comments", "-")
    notes = ""

    # If part number is present
    if part_no:
        part_description = "Already existing in ERP"
        
    # Exception handling for `part_no` when it's missing or "TBD" and `counter_str` is provided.
    try:
        if (not part_no or part_no.strip().upper() == "TBD") and counter_str:  # Only generate if not present

            if component_name == "PCB Name":
                # Final output part number format B14-NEW-01+ 
                # Part Description format: For Base PCB FR4 L x W x substrate thickness
                part_no = f"B14-NEW-{counter_str}+"
                length = case_style_data.get('caseDimensions', {}).get('length')
                width = case_style_data.get('caseDimensions', {}).get('width')

                # If length or width is missing, default to 'L' or 'W'
                length = length if length else 'L'
                width = width if width else 'W'

                if item.get("name") == "Base PCB":
                    part_description = (
                        f"For Base PCB\n FR4 {length} x {width} x substrate thickness\n"
                    )

                elif item.get("name") == "Coupling PCB":
                    part_description = (
                        f"For Coupling PCB\n FR4 {length} x {width} x substrate thickness\n"
                    )

                elif item.get("name") == "Other PCB":
                    part_description = (
                        f"For Other PCB\n FR4 {length} x {width} x substrate thickness\n"
                    )

                else:
                    part_description = (
                        f"For PCB\n FR4 {length} x {width} x substrate thickness\n"
                    )

            
            elif component_name == "CAN Details":
                # Final output part number format B11-NEW-01+
                # Part Description format: METAL CAN L x W x H or PLASTIC CAN L x W x H
                part_no = f"B11-NEW-{counter_str}+"

                can_material = item.get("canMaterial", "").strip().lower()
                custom_can_material = item.get("customCanMaterial", "").strip()

                if can_material == "metal":
                    part_description = "METAL CAN L x W x H "

                elif can_material == "plastic":
                    part_description = "PLASTIC CAN L x W x H "

                elif can_material == "ceramic":
                    part_description = "CERAMIC CAN L x W x H "

                elif can_material == "others" and custom_can_material:
                    part_description = f"{custom_can_material.capitalize()} CAN L x W x H "

                else:
                    part_description = "CAN L x W x H "

                comments = "Only for type closed"

            elif component_name == "Chip Capacitors":
                # Final output part number format B55-NEW-01+
                # Part Description format: Supplier: {supplier_name} | B-P/N: {supplier_number}
                part_no = f"B55-NEW-{counter_str}+"
                supplier_name = item.get('supplierName', '').strip()
                supplier_number = item.get('supplierNumber', '').strip()

                part_description = f"Supplier: {supplier_name} | B-P/N: {supplier_number}" if supplier_name or supplier_number else "Supplier name, Supplier B-P/N"

            elif component_name == "Chip Inductors":
                # Final output part number format B60-NEW-01+
                # Part Description format: Supplier: {supplier_name} | B-P/N: {supplier_number}
                part_no = f"B65-NEW-{counter_str}+"
                supplier_name = item.get('supplierName', '').strip()
                supplier_number = item.get('supplierNumber', '').strip()

                part_description = f"Supplier: {supplier_name} | B-P/N: {supplier_number}" if supplier_name or supplier_number else "Supplier name, Supplier B-P/N"

            elif component_name == "Chip Resistor":
                # Final output part number format B50-NEW-01+
                # Part Description format: Supplier: {supplier_name} | B-P/N: {supplier_number}
                part_no = f"B50-NEW-{counter_str}+"
                supplier_name = item.get('supplierName', '').strip()
                supplier_number = item.get('supplierNumber', '').strip()

                part_description = f"Supplier: {supplier_name} | B-P/N: {supplier_number}" if supplier_name or supplier_number else "Supplier name, Supplier B-P/N"


            elif component_name == "Transformer":
                # Final output part number format B64-CORECODEEDUEDU1234-01+
                # Part Description format: XFMR 1#wire_gauge RD turns orientation RoHS
                # or XFMR 2 CORES 1#wire_gauge GR turns + 1#wire_gauge GD turns orientation RoHS

                core_bpns = item.get("coreBPN", [])
                edu_code = edu_number or opu_number or "XXXX"

                if core_bpns:
                    first_core = core_bpns[0]
                    if "-" in first_core and "+" in first_core:
                        core_code = first_core.split("-")[1].replace("+", "")
                    else:
                        core_code = first_core  # fallback if format is unexpected
                else:
                    core_code = "HAA"  # Default core code if none found

                part_no = f"B64-{core_code}EDU{edu_code}-{counter_str}+"

                core_type = item.get("coreType", "single").lower()
                wire_type = item.get("wireType", "single").lower()
                orientation = item.get("orientation", "P0")
                wire_configs = item.get("wireGaugeConfig", [])
                number_of_wires = item.get("numberOfWires", 1)

                xfmr_prefix = "XFMR"
                rohs = "RoHS"

                # Case 1: Single Core, Single Wire
                if core_type == "single" and wire_type == "single" and len(wire_configs) == 1:
                    wire_gauge = wire_configs[0].get("gauge", "")
                    turns = wire_configs[0].get("turns", 0)
                    turns = f"{turns + 0.5}T"
                    part_description = f"{xfmr_prefix} 1#{wire_gauge} RD {turns} {orientation} {rohs}"

                # Case 2: Single Core, Two Single Wires
                elif core_type == "single" and wire_type == "single" and len(wire_configs) == 2:
                    wire_gauge_1 = wire_configs[0].get("gauge", "")
                    turns_1 = f"{wire_configs[0].get('turns', 0) + 0.5}T"

                    wire_gauge_2 = wire_configs[1].get("gauge", "")
                    turns_2 = f"{wire_configs[1].get('turns', 0) + 0.5}T"

                    part_description = (
                        f"{xfmr_prefix} 1#{wire_gauge_1} GR {turns_1} + 1#{wire_gauge_2} GD {turns_2} {orientation} {rohs}"
                    )

                # Case 3: Single Core, Twisted Wire
                elif core_type == "single" and wire_type == "twisted":
                    wire_gauge = wire_configs[0].get("gauge", "")
                    turns = f"{wire_configs[0].get('turns', 0) + 0.5}T"
                    part_description = f"{xfmr_prefix} {number_of_wires}#{wire_gauge} RDGR {turns} {orientation} {rohs}"

                # Case 4: Double Core, Single Wire
                elif core_type == "double" and wire_type == "single" and len(wire_configs) == 1:
                    wire_gauge = wire_configs[0].get("gauge", "")
                    turns = f"{wire_configs[0].get('turns', 0) + 0.5}T"
                    part_description = f"{xfmr_prefix} 2 CORES 1#{wire_gauge} RD {turns} {orientation} {rohs}"

                # Case 5: Double Core, Two Single Wires
                elif core_type == "double" and wire_type == "double" and len(wire_configs) == 2:
                    wire_gauge_1 = wire_configs[0].get("gauge", "")
                    turns_1 = f"{wire_configs[0].get('turns', 0) + 0.5}T"

                    wire_gauge_2 = wire_configs[1].get("gauge", "")
                    turns_2 = f"{wire_configs[1].get('turns', 0) + 0.5}T"

                    part_description = (
                        f"{xfmr_prefix} 2 CORES 1#{wire_gauge_1} GR {turns_1} + 1#{wire_gauge_2} GD {turns_2} {orientation} {rohs}"
                    )

                # Case 6: Double Core, Twisted Wires
                elif core_type == "double" and wire_type == "twisted":
                    wire_gauge = wire_configs[0].get("gauge", "")
                    turns = f"{wire_configs[0].get('turns', 0) + 0.5}T"
                    part_description = f"{xfmr_prefix} 2 CORES {number_of_wires}#{wire_gauge} RDGR {turns} {orientation} {rohs}"

                else:
                    part_description = f"{xfmr_prefix} Core (B60), wire gauge, number of turns, {orientation} {rohs}"

                comments = "-"
                notes = "-"

            elif component_name == "Resonator Details":
                # Final output part number format B51-18-09-2345-1+
                # Part Description format: RESONATOR WITH SOLDERED TAB or RESONATOR WITHOUT TAB
                resonator_size = item.get("resonatorSize", "").strip()
                dielectric_constant = item.get("dielectricConstant", "").strip() or "09"
                resonator_frequency = item.get("resonatorFrequency", "").strip() or (edu_number or "2345")
                assembly_type = item.get("assemblyType", "").strip().lower()

                # Mapping resonator size to component number
                size_to_component_number = {
                    "1.5": "B51-18-",
                    "2": "B51-29-",
                    "3": "B51-16-",
                    "4": "B51-15-",
                    "5": "B51-17-",
                    "6": "B51-22-",
                    "12": "B51-24-",
                }

                component_number = size_to_component_number.get(resonator_size, "B51-15-")  # Default to B51-15- if not found

                part_no = f"{component_number}{dielectric_constant}-{resonator_frequency}-{int(counter_str)}+"

                if assembly_type == "tab":
                    part_description = "RESONATOR WITH SOLDERED TAB"
                elif assembly_type == "wire":
                    part_description = "RESONATOR WITHOUT TAB"
                else:
                    part_description = "RESONATOR"

                comment = item.get("comments")
                resonator = item.get("resonatorLength")

                if comment and resonator:
                    comments = str(comment) + " , Length: " + str(resonator)
                elif comment:
                    comments = str(comment)
                elif resonator:
                    comments = "Length: " + str(resonator)
                else:
                    comments = "-"


            elif component_name == "Air Coil":
                # Final output part number format B65-NEW-01+
                # Part Description format: AIRCOIL {wire_gauge} AWG, ID {inner_diameter}, {number_of_turns}T, Length {length}, Width {width}
                part_no = f"B65-NEW-{counter_str}+"

                wire_gauge = item.get('wireGauge', '').strip()
                inner_diameter = item.get('innerDiameter', '').strip()
                number_of_turns = item.get('numberOfTurns', '').strip()
                length = item.get('lengthOfAircoil', '').strip()
                width = item.get('widthOfAircoil', '').strip()

                if wire_gauge and inner_diameter and number_of_turns and length and width:
                    part_description = (
                        f"AIRCOIL {wire_gauge} AWG, ID {inner_diameter}, {number_of_turns}T, "
                        f"Length {length}, Width {width}"
                    )
                else:
                    part_description = "Wire gauge, inner diameter, number of turns, length, width"

                # Optional: you can also handle comments for air coil if needed


            elif component_name == "Shield":
                # Final output part number format B17-EDU1234-01+
                # Part Description format: ----
                part_no = f"B17-EDU{edu_number or 'YYYY'}-{counter_str}+"
                part_description = "----"
                # notes = "SHIELD 1.145 X .230 X .010 THK"
                notes = ""

            elif component_name == "Finger Details":
                # Final output part number format B85-EDU1234-01+
                # Part Description format: ----
                edu_code = edu_number or "XXX"

                # Part number format using only EDU number
                part_no = f"B85-EDU{edu_code}-{counter_str}+"

                # Part description as per your requirement
                part_description = "----"

                # Fixed comment for Finger Details
                # comments = "Part to be finalized by CAD."
            
            elif component_name == "Copper Flap Details":
                # Final output part number format B85-TMEDU1234-01+
                # Part Description format: ----
                edu_code = edu_number or "XXX"

                # Part number format
                part_no = f"B85-TMEDU{edu_code}-{counter_str}+"

                # Part description format
                part_description = "----"

                # Fixed comment
                # comments = "Part to be finalized by CAD."
                # notes = "COPPER TAB .08X.03X.003 RoHS"
                notes = ""
            
            elif component_name == "LTCC Details":
                # Final output part number format B52-NEW-01+
                # Part Description format: LTCC 1#wire_gauge RD turns orientation RoHS
                modelName = item.get("modelName") or "XXX"
                part_no = f"Model Name : {modelName}"
                part_description = "-----"
    except Exception as e:
        # Log the exception for debugging purposes
        print(f"Error in _make_component_row for component {component_name}: {e}")
        # Optionally, you can re-raise the exception or set default values
        # For this context, we will continue with potentially incomplete data
        pass

    name_value = item.get("name", "")
    return {
        "id": str(s_no),
        "sNo": s_no,
        "componentName": component_name,
        "component": name_value or component_name,
        "partNo": part_no,
        "rev": item.get("rev", "") or "------",
        "partDescription": item.get("partDescription", "") or part_description ,
        "qtyPerUnit": item.get("qtyPerUnit", "") or 1,
        "stdPkgQty": item.get("stdPkgQty", "") or 1,
        "qtyRequired": item.get("qtyRequired", "") or 1,
        "issuedQty": item.get("issuedQty", ""),
        "qNo": "",
        "comments": comments,
        "notes": notes,
    }


class MakeBillRecordSerializer(serializers.ModelSerializer):
    """
    Serializer for the MakeBillRecord model.
    Handles the serialization and deserialization of MakeBillRecord instances,
    including a UUID-based record_id and JSON-formatted special requirements.
    """
    record_id = serializers.SerializerMethodField()
    special_requirements = serializers.JSONField(required=False)

    class Meta:
        model = MakeBillRecord
        fields = '__all__'

    def get_record_id(self, obj):
        """
        Generates a UUID (version 5) for the record_id using the DNS namespace and the MakeBillRecord's ID.

        Args:
            obj (MakeBillRecord): The MakeBillRecord instance.

        Returns:
            str: A UUID string representing the record_id.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f'MakeBill-{obj.id}'))

