import uuid
from rest_framework import serializers
from pibase.models import PiBaseRecord, PiBaseComponent
from .models import MakeBillRecord, MakeBillStatus


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
    componuntsData = serializers.SerializerMethodField()

    def get_pibaseId(self, obj):
        return obj.id

    def get_componuntsData(self, obj):
        components = obj.components.all() if hasattr(obj.components, "all") else obj.components
        selected_fields = [
            COMPONENT_TO_FIELD_MAP.get(comp.name) for comp in components
            if COMPONENT_TO_FIELD_MAP.get(comp.name)
        ]

        combined_data = []
        s_no = 1
        case_style_data = obj.case_style_data or {}
        edu_number = obj.edu_number
        opu_number = obj.opu_number

        component_counters = {}

        for comp in components:
            field = COMPONENT_TO_FIELD_MAP.get(comp.name)
            if not field:
                continue

            details = getattr(obj, field, None)
            if not details:
                continue

            if isinstance(details, dict):
                for v in details.values():
                    if isinstance(v, list):
                        for item in v:
                            combined_data.append(
                                _make_component_row(item, s_no, comp.name, case_style_data, edu_number, opu_number, component_counters)
                            )
                            s_no += 1
                    elif isinstance(v, dict):
                        combined_data.append(
                            _make_component_row(v, s_no, comp.name, case_style_data, edu_number, opu_number, component_counters)
                        )
                        s_no += 1
                if not any(isinstance(v, (list, dict)) for v in details.values()):
                    combined_data.append(
                        _make_component_row(details, s_no, comp.name, case_style_data, edu_number, opu_number, component_counters)
                    )
                    s_no += 1

            elif isinstance(details, list):
                for item in details:
                    combined_data.append(
                        _make_component_row(item, s_no, comp.name, case_style_data, edu_number, opu_number, component_counters)
                    )
                    s_no += 1

            elif isinstance(details, dict):
                combined_data.append(
                    _make_component_row(details, s_no, comp.name, case_style_data, edu_number, opu_number, component_counters)
                )
                s_no += 1

        return combined_data

    def to_representation(self, instance):
        componunts_data = self.get_componuntsData(instance)

        # Append caseStyle as a new component row if available
        case_style = "Case Style" if instance.case_style_data else None
        if case_style:
            new_row = {
                "id": str(len(componunts_data) + 1),
                "sNo": len(componunts_data) + 1,
                "component": case_style,
                "partNo": "",
                "rev": "",
                "partDescription": "",
                "qtyPerUnit": "",
                "stdPkgQty": "",
                "qtyRequired": "",
                "issuedQty": "",
                "qNo": "",
                "comments": "",
                "notes": "",
            }
            componunts_data.append(new_row)

        return {
            "pibaseId": self.get_pibaseId(instance),
            "componuntsData": componunts_data,
            "recordId": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"PiBase-{instance.id}")),
            "created_by_name": instance.created_by.get_full_name() if instance.created_by else "",
            "created_at": instance.created_at.isoformat() if instance.created_at else None,
            "updated_at": instance.updated_at.isoformat() if instance.updated_at else None,
            "model_name": instance.model_name,
            "revision_number": instance.revision_number,
            "op_number": instance.op_number,
            "opu_number": instance.opu_number,
            "edu_number": instance.edu_number,
            "case_style_data": instance.case_style_data or {},
            "special_requirements": instance.special_requirements or {},
        }



def _make_component_row(item, s_no, component_name=None, case_style_data=None, edu_number=None, opu_number=None, component_counters=None):
    part_no = item.get("bPartNumber", "").strip()
    part_description = ""
    comments = "Use as is from PiBase"
    notes = ""

    if component_name not in component_counters:
        component_counters[component_name] = 1
    else:
        component_counters[component_name] += 1

    counter_str = str(component_counters[component_name]).zfill(2)

    if not part_no:
        if component_name == "PCB Name":
            part_no = f"B14-NEW-{counter_str}+"
            part_description = (
                f"For Base PCB\nPCB FR4 {case_style_data.get('caseDimensions', {}).get('length', '')} x "
                f"{case_style_data.get('caseDimensions', {}).get('width', '')} x substrate thickness\n"
                f"For Coupling PCB\nPCB FR4 {case_style_data.get('caseDimensions', {}).get('length', '')} x "
                f"{case_style_data.get('caseDimensions', {}).get('width', '')} x substrate thickness\n"
                f"For Multilayer PCB\nMULTILAYER PCB FR4 {case_style_data.get('caseDimensions', {}).get('length', '')} x "
                f"{case_style_data.get('caseDimensions', {}).get('width', '')} x overall thickness"
            )

        elif component_name == "CAN Details":
            part_no = f"B11-NEW-{counter_str}+"
            part_description = (
                "If Material is metal:\nMETAL CAN LxWxH\n"
                "If Material is plastic:\nPLASTIC CAN LxWxH"
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
        "rev": item.get("rev", ""),
        "partDescription": part_description or item.get("partDescription", ""),
        "qtyPerUnit": item.get("qtyPerUnit", ""),
        "stdPkgQty": item.get("stdPkgQty", ""),
        "qtyRequired": item.get("qtyRequired", ""),
        "issuedQty": item.get("issuedQty", ""),
        "qNo": "",
        "comments": "",
        "notes": notes,
    }


class MakeBillRecordSerializer(serializers.ModelSerializer):
    pk = serializers.SerializerMethodField()

    class Meta:
        model = MakeBillRecord
        fields = '__all__'

    def get_pk(self, obj):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f'MakeBill-{obj.id}'))
