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
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"PiBase-{obj.id}"))

    def get_created_by_name(self, obj):
        user = getattr(obj, "created_by", None)
        if user:
            full_name = user.get_full_name()
            return full_name or user.username or str(user)
        return ""


class PiBaseComponentSerializer(serializers.ModelSerializer):
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
    pibaseId = serializers.SerializerMethodField()
    componentsData = serializers.SerializerMethodField()
    pibaseRecord = serializers.SerializerMethodField()

    def get_pibaseId(self, obj):
        return obj.id

    def get_componentsData(self, obj):
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

        # ✅ Count only missing part numbers for counter
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

        # ✅ Build final rows
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
                if not item.get("bPartNumber", "").strip():
                    current_counters[comp.name] += 1
                    counter_str = str(current_counters[comp.name]).zfill(2)

                combined_data.append(
                    _make_component_row(item, s_no, comp.name, case_style_data, edu_number, opu_number, counter_str)
                )
                s_no += 1

        return combined_data

    def get_pibaseRecord(self, obj):
        return PiBaseRecordSerializer(obj).data

    def to_representation(self, instance):
        components_data = self.get_componentsData(instance)

        if instance.case_style_data:
            components_data.append({
                "id": str(len(components_data) + 1),
                "sNo": len(components_data) + 1,
                "component": "Case Style",
                "componentName": "Case Style",
                "partNo": "------",
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
                "partNo": "0",
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
    part_no = item.get("bPartNumber", "").strip()
    part_description = ""
    comments = "-"
    notes = ""

        # ✅ If part number is present
    if part_no:
        part_description = "Already existing in ERP"

    if not part_no and counter_str:  # Only generate if not present
        if component_name == "PCB Name":
            part_no = f"B14-NEW-{counter_str}+"
            part_description = (
                f"For Base PCB\nPCB FR4 {case_style_data.get('caseDimensions', {}).get('length', '0')} x "
                f"{case_style_data.get('caseDimensions', {}).get('width', '0')} x substrate thickness\n"
                f"For Coupling PCB\nPCB FR4 {case_style_data.get('caseDimensions', {}).get('length', '0')} x "
                f"{case_style_data.get('caseDimensions', {}).get('width', '0')} x substrate thickness\n"
                f"For Multilayer PCB\nMULTILAYER PCB FR4 {case_style_data.get('caseDimensions', {}).get('length', '0')} x "
                f"{case_style_data.get('caseDimensions', {}).get('width', '0')} x overall thickness"
            )

        elif component_name == "CAN Details":
            part_no = f"B11-NEW-{counter_str}+"
            part_description = (
                "If Material is metal:\nMETAL CAN L x W x H\n"
                "If Material is plastic:\nPLASTIC CAN L x W x H "
            )
            comments = "ONLY FOR TYPE CLOSED\nUse as is from PiBase"

        elif component_name == "Chip Capacitors":
            part_no = f"B55-NEW-{counter_str}+"
            part_description = "Supplier name, Supplier B-P/N"

        elif component_name == "Chip Inductors":
            part_no = f"B65-NEW-{counter_str}+"
            part_description = "Supplier name, Supplier B-P/N"

        elif component_name == "Chip Resistor":
            part_no = f"B50-NEW-{counter_str}+"
            part_description = "Supplier name, Supplier B-P/N"

        elif component_name == "Transformer":
            core_code = item.get("core", "HAA")
            edu_code = edu_number or opu_number or "XXXX"
            part_no = f"B64-{core_code}EDU{edu_code}-{counter_str}+"
            part_description = (
                "Core (B60), wire gauge, number of turns\n"
                "E.g., XFMR 1#34 RD 5.5T P2 RoHS"
            )

        elif component_name == "Chip Resonator":
            part_no = f"B51-15-09-{edu_number or '2345'}-1+"
            part_description = "RESONATOR WITH SOLDERED TAB"

        elif component_name == "Air Coil":
            part_no = f"B65-NEW-{counter_str}+"
            part_description = "Wire gauge, inner diameter, number of turns"

        elif component_name == "Shield":
            part_no = f"B17-XXX{edu_number or 'YYYY'}-01+"
            part_description = "DISPLAY AS EMPTY"
            notes = "SHIELD 1.145 X .230 X .010 THK"

        elif component_name == "Finger Details":
            part_no = f"B85-{edu_number or 'XXX'}{opu_number or 'YYYY'}-01+"
            comments = "? How does the tool know which number to be picked?"

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
    record_id = serializers.SerializerMethodField()
    special_requirements = serializers.JSONField(required=False)

    class Meta:
        model = MakeBillRecord
        fields = '__all__'

    def get_record_id(self, obj):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f'MakeBill-{obj.id}'))
