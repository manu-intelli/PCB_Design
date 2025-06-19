"""
Views for the pibase_makebill app.

Provides API endpoints for listing PiBaseRecord records for the current user,
with filtering, searching, ordering, and error handling.
Includes Swagger documentation for all endpoints.
"""

from rest_framework import generics, status, filters
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django_filters.rest_framework import DjangoFilterBackend
from django.http import Http404
import uuid
from rest_framework.exceptions import NotFound
from collections import defaultdict
from uuid import UUID

from rest_framework.views import APIView
from rest_framework import status
from .models import MakeBillRecord
from .serializers import MakeBillRecordSerializer

from pibase.views import PiBaseRecordPagination
from pibase.models import PiBaseRecord,PiBaseStatus
from .serializers import PiBaseToMakeBillRecordSerializer, MakeBillRecordGetSerializer
from authentication.custom_permissions import IsAuthorized
from authentication.custom_authentication import CustomJWTAuthentication
from . import make_bill_logs

from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from rest_framework.pagination import PageNumberPagination


class MakeBillRecordPagination(PageNumberPagination):
    """
    Pagination class for PiBaseRecord list endpoints.
    """

    page_size = 10
    page_size_query_param = (
        "page_size"  # optional, allow client to set page size in query param
    )
    max_page_size = 100


class PiBaseToMakeBilRecordListView(generics.ListAPIView):
    """
    API endpoint to list PiBaseRecord records for the current user with status=2.
    Supports filtering, searching, and ordering.
    """

    queryset = PiBaseRecord.objects.all()
    serializer_class = PiBaseToMakeBillRecordSerializer
    pagination_class = MakeBillRecordPagination  # Set your pagination class if needed
    permission_classes = [IsAuthenticated]

    filter_backends = [
        DjangoFilterBackend,
        filters.SearchFilter,
        filters.OrderingFilter,
    ]
    filterset_fields = ["status"]
    search_fields = ["model_name", "op_number", "opu_number", "edu_number"]
    ordering_fields = [
        "id",
        "model_name",
        "op_number",
        "opu_number",
        "edu_number",
        "created_at",
        "revision_number",
    ]
    ordering = ["-created_at"]

    @swagger_auto_schema(
        operation_description="List PiBaseRecord records for the current user with status=2. Supports filtering, searching, and ordering.",
        responses={200: PiBaseToMakeBillRecordSerializer(many=True)},
    )
    def list(self, request, *args, **kwargs):
        """
        Returns a list of PiBaseRecord records for the current user.
        Logs success and error events.
        """
        try:
            response = super().list(request, *args, **kwargs)
            make_bill_logs.info(
                f"Successfully listed PiBaseRecord records for user {request.user}."
            )
            return response
        except Exception as e:
            make_bill_logs.error(f"Error listing PiBaseRecord records: {e}")
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def get_queryset(self):
        """
        Returns PiBaseRecord objects created by the current user and with status=2.
        Handles exceptions and logs errors.
        """
        try:
            return PiBaseRecord.objects.filter(status=2)
        except Exception as e:
            make_bill_logs.error(
                f"Error filtering PiBaseRecord for user {self.request.user}: {e}"
            )
            return PiBaseRecord.objects.none()


class MakeBillRetrieveAPIView(generics.RetrieveAPIView):
    """
    API endpoint to retrieve a PiBaseRecord by UUID.
    """

    serializer_class = MakeBillRecordGetSerializer
    permission_classes = [IsAuthorized]
    authentication_classes = [CustomJWTAuthentication]

    @swagger_auto_schema(
        operation_description="Retrieve a PiBaseRecord by UUID.",
        responses={200: MakeBillRecordGetSerializer()},
    )
    def get(self, request, *args, **kwargs):
        return self.retrieve(request, *args, **kwargs)

    def get_queryset(self):
        return PiBaseRecord.objects.all()

    def get_object(self):
        try:
            record_uuid = self.kwargs.get("record_id")
            for obj in self.get_queryset():
                generated_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, f"PiBase-{obj.id}")
                if str(generated_uuid) == str(record_uuid):
                    make_bill_logs.info(
                        f"Successfully retrieved PiBaseRecord {record_uuid}."
                    )
                    return obj
            make_bill_logs.error("Record not found for retrieve.")
            raise Http404("Record not found")
        except Exception as e:
            make_bill_logs.error(
                f"Error in PiBaseRecordRetrieveAPIView.get_object: {e}"
            )
            raise Http404("Record not found")


class MakeBillGetAPIView(APIView):
    """
    Retrieve a MakeBillRecord by UUID.
    """

    serializer_class = MakeBillRecordGetSerializer
    permission_classes = [IsAuthenticated]
    authentication_classes = [CustomJWTAuthentication]

    @swagger_auto_schema(
        operation_description="Retrieve a MakeBillRecord by UUID.",
        responses={200: MakeBillRecordGetSerializer()},
    )
    def get(self, request, record_id):
        try:
            make_bill_logs.info(
                f"Getting MakeBillRecord API called for: {record_id} -- user: {request.user}"
            )

            # Validate and parse UUID
            try:
                uuid_obj = UUID(record_id, version=5) if len(record_id) == 36 else UUID(record_id)
            except ValueError:
                raise NotFound("Invalid UUID format.")

            make_bill = MakeBillRecord.objects.get(id=uuid_obj)
            serializer = MakeBillRecordGetSerializer(make_bill)

            make_bill_logs.info(f"MakeBillRecord: {serializer.data}")
            return Response(serializer.data, status=status.HTTP_200_OK)

        except MakeBillRecord.DoesNotExist:
            make_bill_logs.error(f"MakeBillRecord not found with UUID: {record_id}")
            return Response(
                {"error": "MakeBillRecord not found."}, status=status.HTTP_404_NOT_FOUND
            )

        except Exception as e:
            make_bill_logs.error(
                f"Exception occurred while retrieving MakeBillRecord: {e}"
            )
            return Response(
                {"error": f"Unexpected error: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class MakeBillListAPIView(APIView):
    @swagger_auto_schema(
        operation_description="List all MakeBillRecord entries.",
        responses={200: MakeBillRecordSerializer(many=True)},
    )
    def get(self, request):
        bills = MakeBillRecord.objects.all()
        serializer = MakeBillRecordSerializer(bills, many=True)
        return Response(serializer.data)


# =====================
# Create API
# =====================
class MakeBillCreateAPIView(APIView):
    @swagger_auto_schema(
        operation_description="Create a new MakeBillRecord from provided component data.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                "componentsData": openapi.Schema(
                    type=openapi.TYPE_ARRAY,
                    items=openapi.Items(type=openapi.TYPE_OBJECT),
                ),
                "pibaseId": openapi.Schema(type=openapi.TYPE_INTEGER),
                # Add other fields as needed
            },
            required=["componentsData", "pibaseId"],
        ),
        responses={201: MakeBillRecordSerializer()},
    )
    def post(self, request):
        data = request.data.copy()
        components = data.get("componentsData", [])

        # ✅ Set default status if not provided
        if "status" not in data or not data["status"]:
            data["status"] = 1

        # ✅ Set default revision_number = 1 if not provided
        if "revision_number" not in data or not data["revision_number"]:
            data["revision_number"] = "1"

        # ✅ Set created_by
        if request.user and request.user.is_authenticated:
            data["created_by"] = request.user.id

        # ✅ Component grouping
        component_field_map = {
            "PCB Name": "pcb_details",
            "CAN Details": "can_details",
            "Chip Capacitors": "chip_capacitor_details",
            "Chip Inductors": "chip_inductor_details",
            "Chip Resistors": "chip_resistor_details",
            "Transformer": "transformer_details",
            "Shield": "shield_details",
            "Finger": "finger_details",
            "Copper Flap Details": "copper_flaps_details",
            "Resonator Details": "resonator_details",
            "LTCC Details": "ltcc_details",
            "Chip Aircoil": "chip_aircoil_details",
            "Case Style": "case_style_data",
        }

        grouped = defaultdict(list)
        for comp in components:
            name = comp.get("componentName", "Unknown")
            grouped[name].append(comp)

        for component_name, field_name in component_field_map.items():
            if component_name in grouped:
                data[field_name] = grouped[component_name]

        data["components"] = grouped

        serializer = MakeBillRecordSerializer(data=data)
        if serializer.is_valid():
            makebill = serializer.save()

            # ✅ Update status of related PiBaseRecord
            pibase_id = data.get("pibaseId")
            if pibase_id:
                try:
                    pibase_record = PiBaseRecord.objects.get(id=pibase_id)
                    pibase_status = PiBaseStatus.objects.get(status_code=3)  # Or whatever field maps to status
                    pibase_record.status = pibase_status
                    pibase_record.save()
                except (PiBaseRecord.DoesNotExist, PiBaseStatus.DoesNotExist):
                    pass  # Optional: log or handle

            return Response(serializer.data, status=201)

        return Response(serializer.errors, status=400)


# =====================
# Update API
# =====================
class MakeBillUpdateAPIView(APIView):
    @swagger_auto_schema(
        operation_description="Update an existing MakeBillRecord with partial data, especially components.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                "componentsData": openapi.Schema(
                    type=openapi.TYPE_ARRAY,
                    items=openapi.Items(type=openapi.TYPE_OBJECT),
                ),
                # Add more fields if needed
            },
            required=["componentsData"],
        ),
        responses={200: MakeBillRecordSerializer()},
    )
    def patch(self, request, record_id):
        try:
            make_bill = None
            for record in MakeBillRecord.objects.all():
                if str(uuid5(NAMESPACE_DNS, f"MakeBill-{record.id}")) == str(record_id):
                    make_bill = record
                    break

            if not make_bill:
                return Response(
                    {"error": "MakeBillRecord not found."},
                    status=status.HTTP_404_NOT_FOUND,
                )
            
                    # ✅ Check if status is Pending (id == 1)
            if make_bill.status.id != 1:
                return Response(
                    {"error": "This MakeBill record cannot be edited because its status is not 'Pending'."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            data = request.data.copy()
            components = data.get("componentsData", [])

            # ✅ Set default status if not provided
            if "status" not in data or not data["status"]:
                data["status"] = 1

            # ✅ Auto-increment revision number
            current_rev = make_bill.revision_number
            try:
                new_rev = str(int(current_rev) + 1)
                data["revision_number"] = new_rev
            except (ValueError, TypeError):
                data["revision_number"] = "1"  # fallback

            # ✅ Set updated_by
            if request.user and request.user.is_authenticated:
                data["updated_by"] = request.user.id

            # Component grouping
            component_field_map = {
                "PCB Name": "pcb_details",
                "CAN Details": "can_details",
                "Chip Capacitors": "chip_capacitor_details",
                "Chip Inductors": "chip_inductor_details",
                "Chip Resistors": "chip_resistor_details",
                "Transformer": "transformer_details",
                "Shield": "shield_details",
                "Finger": "finger_details",
                "Copper Flap Details": "copper_flaps_details",
                "Resonator Details": "resonator_details",
                "LTCC Details": "ltcc_details",
                "Chip Aircoil": "chip_aircoil_details",
                "Case Style": "case_style_data",
            }

            grouped = defaultdict(list)
            for comp in components:
                name = comp.get("componentName", "Unknown")
                grouped[name].append(comp)

            for component_name, field_name in component_field_map.items():
                if component_name in grouped:
                    data[field_name] = grouped[component_name]

            if "componentsData" in data:
                del data["componentsData"]

            data["components"] = grouped

            serializer = MakeBillRecordSerializer(make_bill, data=data, partial=True)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data, status=200)
            return Response(serializer.errors, status=400)

        except Exception as e:
            return Response({"error": str(e)}, status=500)


from uuid import uuid5, NAMESPACE_DNS


class MakeBillDeleteAPIView(APIView):
    @swagger_auto_schema(
        operation_description="DELETE is not allowed for MakeBillRecord.",
        responses={405: "DELETE is not allowed"},
    )
    def delete(self, request, record_id):
        return Response({"error": "DELETE is not allowed"}, status=405)


class MakeBillGetAPIView(APIView):
    """
    Retrieve a MakeBillRecord by custom UUID (based on MakeBill-{id})
    and return dynamic componentsData.
    """

    serializer_class = MakeBillRecordSerializer
    permission_classes = [IsAuthenticated]
    authentication_classes = [CustomJWTAuthentication]

    COMPONENT_FIELDS = [
        "pcb_details",
        "can_details",
        "chip_capacitor_details",
        "chip_inductor_details",
        "chip_aircoil_details",
        "transformer_details",
        "shield_details",
        "finger_details",
        "copper_flaps_details",
        "resonator_details",
        "chip_resistor_details",
        "ltcc_details",
    ]

    FIELD_TO_NAME = {
        "pcb_details": "PCB Name",
        "can_details": "CAN Details",
        "chip_capacitor_details": "Chip Capacitors",
        "chip_inductor_details": "Chip Inductors",
        "chip_aircoil_details": "Air Coil",
        "transformer_details": "Transformer",
        "shield_details": "Shield",
        "finger_details": "Finger Details",
        "copper_flaps_details": "Copper Flap Details",
        "resonator_details": "Resonator Details",
        "chip_resistor_details": "Chip Resistor",
        "ltcc_details": "LTCC Details",
    }

    @swagger_auto_schema(
        operation_description="Retrieve a MakeBillRecord by custom UUID and return all component fields dynamically in componentsData.",
        responses={
            200: openapi.Response(
                description="MakeBillRecord dynamic format",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        "pibaseId": openapi.Schema(type=openapi.TYPE_STRING),
                        "componentsData": openapi.Schema(
                            type=openapi.TYPE_ARRAY,
                            items=openapi.Items(type=openapi.TYPE_OBJECT),
                        ),
                        "recordId": openapi.Schema(type=openapi.TYPE_STRING),
                        "created_by_name": openapi.Schema(type=openapi.TYPE_STRING),
                        "created_at": openapi.Schema(type=openapi.TYPE_STRING),
                        "updated_at": openapi.Schema(type=openapi.TYPE_STRING),
                        "model_name": openapi.Schema(type=openapi.TYPE_STRING),
                        "revision_number": openapi.Schema(type=openapi.TYPE_STRING),
                        "op_number": openapi.Schema(type=openapi.TYPE_STRING),
                        "opu_number": openapi.Schema(type=openapi.TYPE_STRING),
                        "edu_number": openapi.Schema(type=openapi.TYPE_STRING),
                        "case_style_data": openapi.Schema(type=openapi.TYPE_OBJECT),
                        "special_requirements": openapi.Schema(
                            type=openapi.TYPE_OBJECT
                        ),
                    },
                ),
            )
        },
    )
    def get(self, request, record_id):
        try:
            # Reverse match MakeBill-{obj.id} using uuid5
            make_bill = None
            for record in MakeBillRecord.objects.all():
                if str(uuid5(NAMESPACE_DNS, f"MakeBill-{record.id}")) == str(record_id):
                    make_bill = record
                    break

            if not make_bill:
                return Response(
                    {"error": "MakeBillRecord not found."},
                    status=status.HTTP_404_NOT_FOUND,
                )

        except Exception as e:
            make_bill_logs.error(f"Exception while retrieving MakeBillRecord: {e}")
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        components_data = []
        s_no = 1

        for field in self.COMPONENT_FIELDS:
            details = getattr(make_bill, field, None)
            if not details:
                continue

            if isinstance(details, list):
                for item in details:
                    components_data.append(self._make_component_row(item, s_no, field))
                    s_no += 1
            elif isinstance(details, dict):
                if any(isinstance(v, list) for v in details.values()):
                    for v in details.values():
                        if isinstance(v, list):
                            for item in v:
                                components_data.append(
                                    self._make_component_row(item, s_no, field)
                                )
                                s_no += 1
                else:
                    components_data.append(
                        self._make_component_row(details, s_no, field)
                    )
                    s_no += 1

        if make_bill.case_style_data:
            components_data.append(
                {
                    "id": str(s_no),
                    "sNo": s_no,
                    "componentName": "Case Style",
                    "component": "Case Style",
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
            )
            s_no += 1

        response_data = {
            "pibaseId": str(make_bill.id),
            "componentsData": components_data,
            "recordId": str(uuid5(NAMESPACE_DNS, f"MakeBill-{make_bill.id}")),
            "created_by_name": (
                make_bill.created_by.get_full_name() if make_bill.created_by else ""
            ),
            "created_at": (
                make_bill.created_at.isoformat() if make_bill.created_at else ""
            ),
            "updated_at": (
                make_bill.updated_at.isoformat() if make_bill.updated_at else ""
            ),
            "model_name": make_bill.model_name,
            "revision_number": make_bill.revision_number,
            "op_number": make_bill.op_number,
            "opu_number": make_bill.opu_number,
            "edu_number": make_bill.edu_number,
            "case_style_data": make_bill.case_style_data or {},
            "special_requirements": make_bill.special_requirements or {},
            "pibaseRecord": make_bill.pibaseRecord if make_bill.pibaseRecord else "",
        }

        return Response(response_data, status=status.HTTP_200_OK)

    def _make_component_row(self, item, s_no, field):
        component_name = self.FIELD_TO_NAME.get(field, field)
        return {
            "id": str(s_no),
            "sNo": s_no,
            "componentName": component_name,
            "component": item.get("component", component_name),
            "partNo": item.get("partNo", ""),
            "rev": item.get("rev", ""),
            "partDescription": item.get("partDescription", ""),
            "qtyPerUnit": item.get("qtyPerUnit", ""),
            "stdPkgQty": item.get("stdPkgQty", ""),
            "qtyRequired": item.get("qtyRequired", ""),
            "issuedQty": item.get("issuedQty", ""),
            "qNo": item.get("qNo", ""),
            "comments": item.get("comments", ""),
            "notes": item.get("notes", ""),
        }
