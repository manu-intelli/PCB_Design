"""
Views for the pibase_makebill app.

Provides API endpoints for listing PiBaseRecord records for the current user,
with filtering, searching, ordering, and error handling.
Includes Swagger documentation for all endpoints.
"""

from rest_framework import generics, status, filters
from rest_framework.generics import ListAPIView
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
from pibase.models import PiBaseRecord, PiBaseStatus
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
    Defines the page size and allows clients to specify it via a query parameter.
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
        """
        Handles GET requests to retrieve a single PiBaseRecord by its UUID.
        """
        return self.retrieve(request, *args, **kwargs)

    def get_queryset(self):
        """
        Returns the queryset of all PiBaseRecord objects.
        """
        return PiBaseRecord.objects.all()

    def get_object(self):
        """
        Retrieves a single PiBaseRecord object based on the UUID provided in the URL.
        It iterates through all records and generates a UUID for each to find a match.
        Raises Http404 if no record is found or an unexpected error occurs during retrieval.
        """
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
        except Http404:
            # Re-raise Http404 as it's an expected outcome for "not found"
            raise
        except Exception as e:
            make_bill_logs.error(
                f"Error in PiBaseRecordRetrieveAPIView.get_object: {e}"
            )
            # Raise a more generic Http404 for unexpected errors during retrieval
            raise Http404("Record not found due to an internal error.")


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
        """
        Handles GET requests to retrieve a MakeBillRecord by its UUID.
        Validates the UUID format and retrieves the corresponding record.
        Logs information and errors during the process.
        """
        try:
            make_bill_logs.info(
                f"Getting MakeBillRecord API called for: {record_id} -- user: {request.user}"
            )

            # Validate and parse UUID
            try:
                uuid_obj = (
                    UUID(record_id, version=5)
                    if len(record_id) == 36
                    else UUID(record_id)
                )
            except ValueError as ve:
                make_bill_logs.error(f"Invalid UUID format provided: {record_id} - {ve}")
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
        except NotFound as nf:
            return Response(
                {"error": str(nf)}, status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            make_bill_logs.error(
                f"Exception occurred while retrieving MakeBillRecord: {e}"
            )
            return Response(
                {"error": f"Unexpected error: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class MakeBillListAPIView(ListAPIView):
    """
    API endpoint to list all MakeBillRecord entries with pagination.
    """
    queryset = MakeBillRecord.objects.all()
    serializer_class = MakeBillRecordSerializer
    permission_classes = [IsAuthenticated]
    pagination_class = MakeBillRecordPagination

    @swagger_auto_schema(
        operation_description="List all MakeBillRecord entries with pagination.",
        responses={200: MakeBillRecordSerializer(many=True)},
    )
    def get(self, request, *args, **kwargs):
        """
        Handles GET requests to list all MakeBillRecord entries.
        """
        return super().get(request, *args, **kwargs)


# =====================
# Create API
# =====================
from collections import defaultdict


class MakeBillCreateAPIView(APIView):
    """
    API endpoint to create a new MakeBillRecord from provided component data.
    """
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
                # Add other fields like model_name, edu_number, etc.
            },
            required=["componentsData", "pibaseId"],
        ),
        responses={201: MakeBillRecordSerializer()},
    )
    def post(self, request):
        """
        Handles POST requests to create a new MakeBillRecord.
        Processes component data, sets default values, and updates the status of the associated PiBaseRecord.
        """
        try:
            data = request.data.copy()
            components = data.get("componentsData", [])

            # Set defaults
            data.setdefault("status", 1)
            data.setdefault("revision_number", "1")
            if request.user and request.user.is_authenticated:
                data["created_by"] = request.user.id

            # Component Field Mapping
            component_field_map = {
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

            # Grouping components
            grouped = defaultdict(list)
            others = []

            for comp in components:
                name = comp.get("componentName", "").strip()
                if name in component_field_map:
                    grouped[name].append(comp)
                else:
                    others.append(comp)

            # Assign mapped fields
            for component_name, field_name in component_field_map.items():
                if component_name in grouped:
                    data[field_name] = grouped[component_name]

            # Save full components + unknown to 'components' and 'others'
            data["components"] = grouped
            if others:
                data["others"] = others

            # Serialize and save
            serializer = MakeBillRecordSerializer(data=data)
            if serializer.is_valid():
                makebill = serializer.save()

                # Update PiBase status
                pibase_id = data.get("pibaseId")
                if pibase_id:
                    try:
                        pibase_record = PiBaseRecord.objects.get(id=pibase_id)
                        pibase_status = PiBaseStatus.objects.get(status_code=3)
                        pibase_record.status = pibase_status
                        pibase_record.save()
                    except (PiBaseRecord.DoesNotExist, PiBaseStatus.DoesNotExist) as e:
                        make_bill_logs.warning(f"Could not update PiBaseRecord status for id {pibase_id}: {e}")
                    except Exception as e:
                        make_bill_logs.error(f"Unexpected error updating PiBaseRecord status for id {pibase_id}: {e}")
                        # Depending on desired behavior, you might re-raise or return an error here
                        pass

                return Response(serializer.data, status=201)

            return Response(serializer.errors, status=400)
        except Exception as e:
            make_bill_logs.error(f"Error creating MakeBillRecord: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# =====================
# Update API
# =====================
class MakeBillUpdateAPIView(APIView):
    """
    API endpoint to update an existing MakeBillRecord with partial data, especially components.
    """
    @swagger_auto_schema(
        operation_description="Update an existing MakeBillRecord with partial data, especially components.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                "componentsData": openapi.Schema(
                    type=openapi.TYPE_ARRAY,
                    items=openapi.Items(type=openapi.TYPE_OBJECT),
                ),
                # Add additional fields here if needed
            },
            required=["componentsData"],
        ),
        responses={200: MakeBillRecordSerializer()},
    )
    def patch(self, request, record_id):
        """
        Handles PATCH requests to update an existing MakeBillRecord.
        Allows partial updates, specifically for component data, and updates revision number.
        """
        try:
            make_bill = None
            # Iterate to find the MakeBillRecord by its generated UUID
            for record in MakeBillRecord.objects.all():
                if str(uuid5(NAMESPACE_DNS, f"MakeBill-{record.id}")) == str(record_id):
                    make_bill = record
                    break

            if not make_bill:
                return Response(
                    {"error": "MakeBillRecord not found."},
                    status=status.HTTP_404_NOT_FOUND,
                )

            # Check if the record status allows updates
            if not make_bill.status or make_bill.status.id != 1:
                return Response(
                    {"error": "Only records with 'Pending' status can be updated."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            data = request.data.copy()
            components = data.get("componentsData", [])

            # Set status and revision
            data.setdefault("status", 1)
            try:
                current_rev = int(make_bill.revision_number or 0)
                data["revision_number"] = str(current_rev + 1)
            except (ValueError, TypeError) as e:
                make_bill_logs.warning(f"Error converting revision number for MakeBillRecord {make_bill.id}: {e}. Setting to '1'.")
                data["revision_number"] = "1"

            # Set updated_by
            if request.user and request.user.is_authenticated:
                data["updated_by"] = request.user.id

            # Field mapping for components
            component_field_map = {
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

            grouped = defaultdict(list)
            others = []

            for comp in components:
                name = comp.get("componentName", "").strip()
                if name in component_field_map:
                    grouped[name].append(comp)
                else:
                    others.append(comp)

            # Assign to mapped fields
            for component_name, field_name in component_field_map.items():
                if component_name in grouped:
                    data[field_name] = grouped[component_name]

            # Save unknown components
            if others:
                data["others"] = others

            # Remove raw componentsData as it's processed into other fields
            data["components"] = grouped
            data.pop("componentsData", None)

            serializer = MakeBillRecordSerializer(make_bill, data=data, partial=True)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data, status=200)

            return Response(serializer.errors, status=400)

        except Exception as e:
            make_bill_logs.error(f"Error updating MakeBillRecord {record_id}: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        

from uuid import uuid5, NAMESPACE_DNS


class MakeBillDeleteAPIView(APIView):
    """
    API endpoint for deleting MakeBillRecord.
    DELETE method is explicitly not allowed for this resource.
    """
    @swagger_auto_schema(
        operation_description="DELETE is not allowed for MakeBillRecord.",
        responses={405: "DELETE is not allowed"},
    )
    def delete(self, request, record_id):
        """
        Handles DELETE requests. Returns a 405 Method Not Allowed response.
        """
        return Response({"error": "DELETE is not allowed"}, status=405)


class MakeBillGetAPIView(APIView):
    """
    Retrieve a MakeBillRecord by custom UUID (based on MakeBill-{id})
    and return dynamic componentsData including 'others' and case style.
    """

    serializer_class = None  # Optional
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
                        "pibaseRecord": openapi.Schema(type=openapi.TYPE_OBJECT),
                    },
                ),
            )
        },
    )
    def get(self, request, record_id):
        """
        Handles GET requests to retrieve a MakeBillRecord by its custom UUID.
        Dynamically extracts and structures component data from various fields
        (e.g., pcb_details, can_details, others) into a unified 'componentsData' list.
        """
        try:
            make_bill = None
            for record in MakeBillRecord.objects.all():
                if str(uuid5(NAMESPACE_DNS, f"MakeBill-{record.id}")) == str(record_id):
                    make_bill = record
                    break

            if not make_bill:
                make_bill_logs.error(f"MakeBillRecord not found for UUID: {record_id}")
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
        auto_sno = 1

        # Handle known fields
        for field in self.COMPONENT_FIELDS:
            try:
                details = getattr(make_bill, field, None)
                if not details:
                    continue

                if isinstance(details, list):
                    for item in details:
                        s_no = item.get("sNo") or auto_sno
                        components_data.append(self._make_component_row(item, s_no, field))
                        auto_sno = max(auto_sno, s_no + 1) if isinstance(s_no, int) else auto_sno + 1

                elif isinstance(details, dict):
                    if any(isinstance(v, list) for v in details.values()):
                        for v in details.values():
                            if isinstance(v, list):
                                for item in v:
                                    s_no = item.get("sNo") or auto_sno
                                    components_data.append(self._make_component_row(item, s_no, field))
                                    auto_sno = max(auto_sno, s_no + 1) if isinstance(s_no, int) else auto_sno + 1
                    else:
                        s_no = details.get("sNo") or auto_sno
                        components_data.append(self._make_component_row(details, s_no, field))
                        auto_sno = max(auto_sno, s_no + 1) if isinstance(s_no, int) else auto_sno + 1
            except Exception as e:
                make_bill_logs.error(f"Error processing component field '{field}' for MakeBillRecord {make_bill.id}: {e}")
                # Continue processing other fields even if one fails

        # Handle 'others' field
        try:
            others = getattr(make_bill, "others", [])
            if isinstance(others, list):
                for item in others:
                    s_no = item.get("sNo") or auto_sno
                    components_data.append(self._make_component_row(item, s_no, "others"))
                    auto_sno = max(auto_sno, s_no + 1) if isinstance(s_no, int) else auto_sno + 1
        except Exception as e:
            make_bill_logs.error(f"Error processing 'others' field for MakeBillRecord {make_bill.id}: {e}")
            # Continue with the rest of the response

        response_data = {
            "pibaseId": str(make_bill.id),
            "componentsData": components_data,
            "recordId": str(uuid5(NAMESPACE_DNS, f"MakeBill-{make_bill.id}")),
            "created_by_name": make_bill.created_by.get_full_name() if make_bill.created_by else "",
            "created_at": make_bill.created_at.isoformat() if make_bill.created_at else "",
            "updated_at": make_bill.updated_at.isoformat() if make_bill.updated_at else "",
            "model_name": make_bill.model_name,
            "revision_number": make_bill.revision_number,
            "op_number": make_bill.op_number,
            "opu_number": make_bill.opu_number,
            "edu_number": make_bill.edu_number,
            "pibaseRecord": make_bill.pibaseRecord if make_bill.pibaseRecord else "",
        }

        return Response(response_data, status=status.HTTP_200_OK)

    def _make_component_row(self, item, s_no, field):
        """
        Helper function to construct a single row for a component within the `componentsData` list.

        Args:
            item (dict): The dictionary containing the raw details of the component.
            s_no (int): The serial number for the component row.
            field (str): The name of the field from which this component data originated (e.g., "pcb_details").

        Returns:
            dict: A dictionary representing a single structured component row.
        """
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