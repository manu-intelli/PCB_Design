from django.http import Http404
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from drf_yasg import openapi
from authentication.custom_permissions import IsAuthorized
from authentication.custom_authentication import CustomJWTAuthentication
from .models import CADDesignTemplates, CADVerifierTemplates, CADApproverTemplates
from .services import get_categories_for_component_id, create_cad_template,\
    get_sub_categories_two_for_subcategory_id,  get_design_options_for_sub_category,get_design_rules_for_design_option,\
    get_verifier_fields_by_params, create_cad_verifier_template, compare_verifier_data_with_rules_and_designs, get_verifier_record, \
    save_approver_results
from drf_yasg.utils import swagger_auto_schema
from .serializers import CADDesignTemplatesSerializer
from . import right_to_draw_logs
from django.core.exceptions import ObjectDoesNotExist


class ComponentAPIView(APIView):
    permission_classes = [IsAuthorized]
    authentication_classes = [CustomJWTAuthentication]
    
    def get(self, request, component_id):
        is_verifier = int(request.GET.get('is_verifier', 0))
        try:
            right_to_draw_logs.info(f"Get Component API View called for: {component_id}, is_verfier:'{is_verifier}' -- user:{request.user}")            
            response = get_categories_for_component_id(component_id, is_verifier)
            l_response = len(response)
            right_to_draw_logs.info(f"Get Component Data for: {component_id}, is_verfier:{is_verifier} --- No: Categories:{l_response}")
            return Response(response, status=status.HTTP_200_OK) 
        
        except Http404 as e:
            right_to_draw_logs.info(f"Http404 Error in Component API View for component_id: {component_id} -- user: {request.user} -- {str(e)}")
            right_to_draw_logs.error(f"Http404 Error in Component API View for component_id: {component_id} -- user: {request.user} -- {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_404_NOT_FOUND)
        
        except Exception as e:
            error_log = f"Exception Occurred in Component API View for component_id: {component_id} -- user: {request.user} -- {str(e)}"
            right_to_draw_logs.info(error_log)
            right_to_draw_logs.error(error_log)
            return Response({"error": f"Exception Occurred {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SubCategoryTwoAPIView(APIView):
    permission_classes = [IsAuthorized]
    authentication_classes = [CustomJWTAuthentication]
    def get(self, request, sub_category_id):
        try:
            right_to_draw_logs.info(f"Get Sub Category Two API View called for: {sub_category_id} -- user: {request.user}")
            response =  get_sub_categories_two_for_subcategory_id(sub_category_id)
            l_response = len(response.data)
            right_to_draw_logs.info(f"Get Sub Category Two Data for: {sub_category_id} -- No: Categories:{l_response}")
            return Response(response.data, status=status.HTTP_200_OK)
        except Http404 as e:
            error_log = f"Http404 Error in Sub Category Two API View for sub_category_id: {sub_category_id} -- user: {request.user} -- {str(e)}"          
            right_to_draw_logs.info(error_log)
            right_to_draw_logs.error(error_log)
            return Response({"error": str(e)}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            error_log = f"Exception Occurred in Sub Category Two API View for sub_category_id: {sub_category_id} -- user: {request.user} -- {str(e)}"
            right_to_draw_logs.info(error_log)
            right_to_draw_logs.error(error_log)
            return Response({"error": f"Exception Occurred {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class DesignOptionAPIView(APIView):
    permission_classes = [IsAuthorized]
    authentication_classes = [CustomJWTAuthentication]
    def get(self, request, sub_category_id):
        right_to_draw_logs.info(f"Get Design Options API View called for: {sub_category_id} -- user: {request.user}")
        try:
            response =  get_design_options_for_sub_category(sub_category_id)
            l_response = len(response)
            right_to_draw_logs.info(f"Get Design Options Data for: {sub_category_id} -- No: Options:{l_response}")
            return Response(response, status=status.HTTP_200_OK)
        except Http404 as e:            
            error_log = f"Http404 Error in Design Options API View for sub_category_id: {sub_category_id} -- user: {request.user} -- {str(e)}"          
            right_to_draw_logs.info(error_log)
            right_to_draw_logs.error(error_log)
            return Response({"error": str(e)}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            error_log = f"Exception Occurred in Design Options API View for sub_category_id: {sub_category_id} -- user: {request.user} -- {str(e)}"
            right_to_draw_logs.info(error_log)
            right_to_draw_logs.error(error_log)
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class DesignRuleAPIView(APIView):
    permission_classes = [IsAuthorized]
    authentication_classes = [CustomJWTAuthentication]
    def get(self, request):
        try:
            design_option_ids = request.query_params.get('design_option_ids', None)          
            right_to_draw_logs.info(f"Get Design Rule API View called for: {design_option_ids} -- user: {request.user}")     
            if design_option_ids:
                right_to_draw_logs.info(f"Provided Design Option IDs :{design_option_ids}")
                design_option_ids = design_option_ids.split(',')                
                design_option_ids = [int(id.strip()) for id in design_option_ids]
                right_to_draw_logs.info(f"Get Design Rules for: {design_option_ids} -- user: {request.user}")
                response =  get_design_rules_for_design_option(design_option_ids)
                return Response(response, status=status.HTTP_200_OK)
            else:
                error_log = f"Http404 Error: No design_option_ids provided"
                right_to_draw_logs.info(error_log)
                right_to_draw_logs.erro(error_log)    
                raise Http404("No design_option_ids provided")
        except Http404 as e:            
            error_log = f"Http404 Error in Design Rules API View for design_option_ids: {design_option_ids} -- user: {request.user} -- {str(e)}"          
            right_to_draw_logs.info(error_log)
            right_to_draw_logs.error(error_log)
            return Response({"error": str(e)}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            error_log = f"Exception Occurred in Design Rules API View for design_option_ids: {design_option_ids} -- user: {request.user} -- {str(e)}"
            right_to_draw_logs.info(error_log)
            right_to_draw_logs.error(error_log)
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class CADDesignTemplatesAPIView(APIView):
    permission_classes = [IsAuthorized]
    authentication_classes = [CustomJWTAuthentication]
    
    def get(self, request,  id, *args, **kwargs):                
        #Create a log for the API call
        right_to_draw_logs.info(f"Get CAD Design Templates API View called for: {id} -- user: {request.user}") 
        if id:
            right_to_draw_logs.info(f"If CAD Template ID Provided: {id}")
            try:                
                cad_template = CADDesignTemplates.objects.get(id=id)                
                serializer = CADDesignTemplatesSerializer(cad_template)
                return Response(serializer.data, status=status.HTTP_200_OK)
            except CADDesignTemplates.DoesNotExist:
                error_log = f"CADDesign Templates Does Not Exist For CAD Template ID: {id}"
                right_to_draw_logs.info(error_log)
                right_to_draw_logs.error(error_log)
                return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        
        else:
            right_to_draw_logs.info(f"If CAD Template ID Not Provided")
            cad_templates = CADDesignTemplates.objects.all()            
            serializer = CADDesignTemplatesSerializer(cad_templates, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)


    @swagger_auto_schema(
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'oppNumber': openapi.Schema(type=openapi.TYPE_STRING, description='Opportunity Number'),
                'opuNumber': openapi.Schema(type=openapi.TYPE_STRING, description='Opu Number'),
                'eduNumber': openapi.Schema(type=openapi.TYPE_STRING, description='Education Number'),
                'modelName': openapi.Schema(type=openapi.TYPE_STRING, description='Model Name'),
                'partNumber': openapi.Schema(type=openapi.TYPE_STRING, description='Part Number'),
                'revisionNumber': openapi.Schema(type=openapi.TYPE_STRING, description='Revision Number'),
                'component': openapi.Schema(type=openapi.TYPE_INTEGER, description='Component ID (e.g., b14)'),
                'componentSpecifications': openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    additional_properties=openapi.Schema(type=openapi.TYPE_STRING, description="Dynamic specification fields")
                ),
                'designOptions': openapi.Schema(
                    type=openapi.TYPE_ARRAY,
                    items=openapi.Items(type=openapi.TYPE_STRING),
                    description='Design options (array of strings)'
                ),
                'secondarySubLevel': openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    additional_properties=openapi.Schema(type=openapi.TYPE_STRING, description="Secondary sub level fields")
                ),
                'remarks': openapi.Schema(type=openapi.TYPE_STRING, description='Remarks')
            },
            required=['oppNumber', 'opuNumber', 'modelName', 'partNumber', 'component'],
        ),
        responses={201: 'Template Created', 400: 'Bad Request'}
    )
    def post(self, request):
        #Create a log for the API call
        right_to_draw_logs.info(f"Post CAD Design Templates API View called -- user: {request.user}")
        user = request.user
        template, error = create_cad_template(request.data, user)        
        if error:
            error_log=f"Error in Creating Template: {error}"
            right_to_draw_logs.info(error_log)
            right_to_draw_logs.error(error_log)
            return Response(error, status=status.HTTP_400_BAD_REQUEST)
                
        return Response(template.id, status=status.HTTP_201_CREATED)


class MstVerifierFieldFilterAPIView(APIView):
    permission_classes = [IsAuthorized]
    authentication_classes = [CustomJWTAuthentication]

    def get(self, request):        
        component_id = request.query_params.get('component_id', None)
        category_id = request.query_params.get('category_id', None)
        sub_category_id = request.query_params.get('sub_category_id', None)
        right_to_draw_logs.info(f"Get Verifier Fields API View called for: {component_id}, {category_id}, {sub_category_id} -- user: {request.user}")
        try:            
            serialized_data = get_verifier_fields_by_params(
                component_id=component_id,
                category_id=category_id,
                sub_category_id=sub_category_id
            )                        

            return Response(serialized_data, status=status.HTTP_200_OK)

        except Exception as e:
            error_log = f"Exception Occurred in Verifier Fields API View for component_id: {component_id}, category_id: {category_id}, sub_category_id: {sub_category_id} -- user: {request.user} -- {str(e)}"
            right_to_draw_logs.info(error_log)
            right_to_draw_logs.error(error_log)            
            return Response({"error": f"Exception occurred: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class CADVerifierTemplateCreateAPIView(APIView):
    permission_classes = [IsAuthorized]
    authentication_classes = [CustomJWTAuthentication]

    @swagger_auto_schema(
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'oppNumber': openapi.Schema(type=openapi.TYPE_STRING, description='Opp Number'),
                'opuNumber': openapi.Schema(type=openapi.TYPE_STRING, description='Opu Number'),
                'eduNumber': openapi.Schema(type=openapi.TYPE_STRING, description='Edu Number'),
                'modelName': openapi.Schema(type=openapi.TYPE_STRING, description='Model Name'),
                'partNumber': openapi.Schema(type=openapi.TYPE_STRING, description='Part Number'),
                'revisionNumber': openapi.Schema(type=openapi.TYPE_STRING, description='Revision Number'),
                'component': openapi.Schema(type=openapi.TYPE_INTEGER, description='Component ID (e.g., b14)'),
                'componentSpecifications': openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    additional_properties=openapi.Schema(type=openapi.TYPE_STRING, description="Dynamic specification fields")
                ),
                'verifierQueryData': openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    additional_properties=openapi.Schema(type=openapi.TYPE_STRING, description="Verifier query fields")
                ),
                'remarks': openapi.Schema(type=openapi.TYPE_STRING, description='Remarks')
            },
            required=['oppNumber', 'opuNumber', 'modelName', 'partNumber', 'component'],
        ),
        responses={201: 'Template Created', 400: 'Bad Request'}
    )
    def post(self, request):
        right_to_draw_logs.info(f"Post CAD Verifier Templates API View called -- user: {request.user}")
        try:
            user = request.user        
            res = compare_verifier_data_with_rules_and_designs(request.data)
            if not res:
                right_to_draw_logs.info(f"Designer Data does not match with the rules and designs to verify further")
                right_to_draw_logs.error(f"Designer Data does not match with the rules and designs to verify further")
                return Response("Invalid Data, Designer Data is not present for selected Values", status=status.HTTP_400_BAD_REQUEST)
            
            try:
                template, error = create_cad_verifier_template(request.data, user)
            except Exception as e:
                error = f"Exception occurred: {e}"
                right_to_draw_logs.info(error)
                right_to_draw_logs.error(error)
                raise "Exception occurred {e}"
                        
            if error:
                error_log=f"Error in Creating CAD Verifier Template: {error}"
                right_to_draw_logs.info(error_log)
                right_to_draw_logs.error(error_log)
                return Response(error, status=status.HTTP_400_BAD_REQUEST)
            
            return Response({"template_id":template.id, "res":res}, status=status.HTTP_201_CREATED)
        except Exception as e:
            error_log = f"Exception Occurred in CAD Verifier Templates API View -- user: {request.user} -- {str(e)}"
            right_to_draw_logs.info(error_log)
            right_to_draw_logs.error(error_log)            
            return Response({"error": f"Exception occurred: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class MstVerifierFieldResultAPIView(APIView):
    permission_classes = [IsAuthorized]
    authentication_classes = [CustomJWTAuthentication]
    @swagger_auto_schema(
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'oppNumber': openapi.Schema(type=openapi.TYPE_STRING, description='Opp Number'),
                'opuNumber': openapi.Schema(type=openapi.TYPE_STRING, description='Opu Number'),
                'eduNumber': openapi.Schema(type=openapi.TYPE_STRING, description='Edu Number'),
                'modelName': openapi.Schema(type=openapi.TYPE_STRING, description='Model Name'),
                'partNumber': openapi.Schema(type=openapi.TYPE_STRING, description='Part Number'),
                'revisionNumber': openapi.Schema(type=openapi.TYPE_STRING, description='Revision Number'),
                'component': openapi.Schema(type=openapi.TYPE_INTEGER, description='Component ID (e.g., b14)')                
            },
            required=['oppNumber', 'opuNumber', 'modelName', 'partNumber', 'component'],
        ),
        responses={201: 'Results Created', 400: 'Bad Request'}
    )
    def post(self, request):
        right_to_draw_logs.info(f"Post Verifier Fields Result API View called -- user: {request.user}")
        try:            

            verifier_record_data = get_verifier_record(request.data)     
            res = compare_verifier_data_with_rules_and_designs(verifier_record_data)
            if not res:
                right_to_draw_logs.info(f"Designer Data does not match with the rules and designs to verify further")
                right_to_draw_logs.error(f"Designer Data does not match with the rules and designs to verify further")
                return Response("Invalid Data, Designer Data is not present for selected Values", status=status.HTTP_400_BAD_REQUEST)
            
            return Response({"res":res}, status=200)

        except Exception as e:            
            error_log = f"Exception Occurred in Verifier Fields Result API View -- user: {request.user} -- {str(e)}"
            right_to_draw_logs.info(error_log)
            right_to_draw_logs.error(error_log)
            return Response({"error": f"Exception occurred: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ApproverAPIView(APIView):
    permission_classes = [IsAuthorized]
    authentication_classes = [CustomJWTAuthentication]

    def get(self, request):
        oppNumber = request.query_params.get('oppNumber', None)
        opuNumber = request.query_params.get('opuNumber', None)
        eduNumber = request.query_params.get('eduNumber', None)
        modelName = request.query_params.get('modelName', None)
        partNumber = request.query_params.get('partNumber', None)
        revisionNumber = request.query_params.get('revisionNumber', None)
        component = request.query_params.get('component', None)
        
        data = {
            'oppNumber': oppNumber,
            'opuNumber': opuNumber,
            'eduNumber': eduNumber,
            'modelName': modelName,
            'partNumber': partNumber,
            'revisionNumber': revisionNumber,
            'component': component
        }

        log_Str = f"For OppNumber: {oppNumber}, OpuNumber: {opuNumber}, EduNumber: {eduNumber}, ModelName: {modelName}, PartNumber: {partNumber}, RevisionNumber: {revisionNumber}, Component: {component}"
        right_to_draw_logs.info(f"Get Approver Fields API View called for: {log_Str}")

        try:
            try:
                verifier_record_data = get_verifier_record(data)
            except ObjectDoesNotExist as e:
                error_log = f"Object Does Not Exist in Approver API View -- user: {request.user} -- {str(e)} for {log_Str}"
                right_to_draw_logs.info(error_log)
                right_to_draw_logs.error(error_log)
                return Response({"error": f"No Approver Record Found For Given Variables: {str(e)}"}, status=status.HTTP_404_NOT_FOUND)
          
            res = compare_verifier_data_with_rules_and_designs(verifier_record_data)
            if not res:
                right_to_draw_logs.info(f"Designer Data does not match with the rules and designs to verify further")
                right_to_draw_logs.error(f"Designer Data does not match with the rules and designs to verify further")
                return Response("Invalid Data, Designer Data is not present for selected Values", status=status.HTTP_400_BAD_REQUEST)
            return Response({"res":res}, status=200)
        except Exception as e:
            error_log = f"Exception Occurred in Approver API View -- user: {request.user} -- {str(e)} : for {log_Str}"
            right_to_draw_logs.info(error_log)
            right_to_draw_logs.error(error_log)
            return Response({"error": f"Exception occurred: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @swagger_auto_schema(
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'oppNumber': openapi.Schema(type=openapi.TYPE_STRING, description='Opp Number'),
                'opuNumber': openapi.Schema(type=openapi.TYPE_STRING, description='Opu Number'),
                'eduNumber': openapi.Schema(type=openapi.TYPE_STRING, description='Edu Number'),
                'modelName': openapi.Schema(type=openapi.TYPE_STRING, description='Model Name'),
                'partNumber': openapi.Schema(type=openapi.TYPE_STRING, description='Part Number'),
                'revisionNumber': openapi.Schema(type=openapi.TYPE_STRING, description='Revision Number'),
                'component': openapi.Schema(type=openapi.TYPE_INTEGER, description='Component ID (e.g., b14)'),
                'componentSpecifications': openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    additional_properties=openapi.Schema(type=openapi.TYPE_STRING, description="Dynamic specification fields")
                ),
                'approverQueryData': openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    additional_properties=openapi.Schema(type=openapi.TYPE_STRING, description="Verifier query fields")
                ),
                "status": openapi.Schema(type=openapi.TYPE_STRING, description="Status of the Approver"),
                "comments": openapi.Schema(type=openapi.TYPE_STRING, description="Comments for the Approver")
            },
            required=['oppNumber', 'opuNumber', 'modelName', 'partNumber', 'component'],
        ),
        responses={201: 'Template Created', 400: 'Bad Request'}
    )
    def post(self, request):
        if request.user.is_authenticated:
            user = request.user  # This should be a CustomUser instance if the user is authenticated
        else:
            return Response({"error": "User is not authenticated"}, status=status.HTTP_401_UNAUTHORIZED)
        right_to_draw_logs.info(f"Post Approver Fields API View called -- user: {request.user}")
        try:
            user = request.user            
            result, err = save_approver_results(request.data, user)
            if err:
                error_log = f"Error in Saving Approver Results: {err}"
                right_to_draw_logs.info(error_log)
                right_to_draw_logs.error(error_log)
                return Response(err, status=status.HTTP_400_BAD_REQUEST)
            
            return Response({"template_id":result.id}, status=200)
        except Exception as e:
            error_log = f"Exception Occurred in Approver API View -- user: {request.user} -- {str(e)}"
            right_to_draw_logs.info(error_log)
            right_to_draw_logs.error(error_log)
            return Response({"error": f"Exception occurred: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class CheckDesignerAndVerifierRecordAPIView(APIView):
    permission_classes = [IsAuthorized]
    authentication_classes = [CustomJWTAuthentication]

    def get(self, request):
        oppNumber = request.query_params.get('oppNumber', None)
        opuNumber = request.query_params.get('opuNumber', None)
        eduNumber = request.query_params.get('eduNumber', None)
        modelName = request.query_params.get('modelName', None)
        partNumber = request.query_params.get('partNumber', None)
        revisionNumber = request.query_params.get('revisionNumber', None)            
        component = request.query_params.get('component', None)

        data = {
            'opp_number': oppNumber,
            'opu_number': opuNumber,
            'edu_number': eduNumber,
            'model_name': modelName,
            'part_number': partNumber,
            'revision_number': revisionNumber,
            'component_Id':component                 
        }

        log_Str = f"For OppNumber: {oppNumber}, OpuNumber: {opuNumber}, EduNumber: {eduNumber}, ModelName: {modelName}, PartNumber: {partNumber}, RevisionNumber: {revisionNumber}, Component: {component}"
        right_to_draw_logs.info(f"Check Designer and Verifier Record API View called for: {log_Str}")

        try:
            designer_record = CADDesignTemplates.objects.filter(**data).exists()
            verifier_record = CADVerifierTemplates.objects.filter(**data).exists()
            approver_record = CADApproverTemplates.objects.filter(**data).exists()


            return Response({"designer_exists": True if designer_record else False,
                            "verifier_exists": True if verifier_record else False,
                            "approver_exists": True if approver_record else False}, 
                            status=status.HTTP_200_OK)
        except Exception as e:
            error_log = f"Exception Occurred in Check Designer and Verifier Record API View -- user: {request.user} -- {str(e)} : for {log_Str}"
            right_to_draw_logs.info(error_log)
            right_to_draw_logs.error(error_log)
            return Response({"error": f"Exception occurred: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class UserCreatedTemplatesView(APIView):
    permission_classes = [IsAuthorized]

    def get(self, request, *args, **kwargs):
        user = request.user

        # Get all the records created by the logged-in user
        design_templates = CADDesignTemplates.objects.filter(created_by=user)
        verifier_templates = CADVerifierTemplates.objects.filter(created_by=user)
        approver_templates = CADApproverTemplates.objects.filter(created_by=user)

        # Combine all the records (or return them in different responses as needed)
        data = {
            'design_templates': self.get_template_data(design_templates),
            'verifier_templates': self.get_template_data(verifier_templates),
            'approver_templates': self.get_template_data(approver_templates),
        }

        return Response(data, status=status.HTTP_200_OK)

    def get_template_data(self, templates):
        """Helper method to format the response data."""
        return [
            {
                'oppNumber': template.opp_number,
                'opuNumber': template.opu_number,
                'eduNumber': template.edu_number,
                'modelName': template.model_name,
                'partNumber': template.part_number,
                'revisionNumber': template.revision_number,
                'component': template.component_Id.id,
            }
            for template in templates
        ]