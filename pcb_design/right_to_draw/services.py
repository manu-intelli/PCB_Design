from django.http import Http404, HttpResponseServerError
from django.db.models import Prefetch
from collections import defaultdict
from rest_framework.response import Response
from rest_framework import status
from django.db.models import Q
from .models import CADVerifierTemplates, CADDesignTemplates
from masters.models import MstComponent, MstSubCategoryTwo, MstDesignOptions, MstSectionRules, \
    MstSectionGroupings, MstVerifierField, MstVerifierField, MstVerifierRules, MstCategory, MstSubCategory, \
    MstConditions
from .serializers import SectionGroupingsSerializer,SubCategoryTwoSerializer, CADDesignTemplatesSerializer, \
    SectionRulesSerializer, MstVerifierFieldSerializer, CADVerifierTemplateSerializer,CADApproverTemplateSerializer
from django.http import Http404
from django.core.exceptions import ObjectDoesNotExist
from authentication.models import CustomUser
from . import right_to_draw_logs
from rest_framework.exceptions import ValidationError
from authentication.serializers import ForgotPasswordSerializer,GetUserSerializer,UpdateUserSerializer



def get_categories_for_component_id(component_id, is_verifier=0):
    right_to_draw_logs.info(f"Get categories for component_id: {component_id}, is_verifier: {is_verifier}")
    try:
        component = MstComponent.objects.prefetch_related(
            'component_categories',
            'component_categories__subcategories'
        ).get(id=component_id)

        if is_verifier == 1:            
            categories = component.component_categories.filter(is_verifier=True)
        else:            
            categories = component.component_categories.all()

        result = []
        
        for category in categories:            
            if is_verifier == 1:                
                subcategories = category.subcategories.filter(is_verifier=True)
            else:                
                subcategories = category.subcategories.all()
                
            subcategory_data = [
                {
                    'id': subcategory.id,
                    'name': subcategory.sub_category_name,
                    'is_design_options_exists': MstDesignOptions.objects.filter(sub_category_id=subcategory).exists(),
                    'is_sub_2_categories_exists': MstSubCategoryTwo.objects.filter(sub_category_id=subcategory).exists(),
                    'has_verifier_fields': MstVerifierField.objects.filter(sub_category=subcategory).exists()
                }
                for subcategory in subcategories
            ]
            right_to_draw_logs.info(f"Number of Subcategories: {len(subcategory_data)} for Category: {category.name}")
            
            result.append({
                'category_id': category.id,
                'category_name': category.category_name,
                'subcategories': subcategory_data
            })
    
        right_to_draw_logs.info(f"Number of Categorie: {len(result)} for Component ID: {component_id}")
        return result
        
    except MstComponent.DoesNotExist:
        error_log = f"Component with ID {component_id} does not exist."
        right_to_draw_logs.info(error_log)
        right_to_draw_logs.error(error_log)
        raise Http404("Component with the given ID does not exist.")


def get_sub_categories_two_for_subcategory_id(sub_category_id):
    right_to_draw_logs.info(f"Get sub-categories two for subcategory_id: {sub_category_id}")
    try:        
        sub_categories_two = MstSubCategoryTwo.objects.filter(sub_category_id=sub_category_id)
        right_to_draw_logs.info(f"Number of Sub-categories two: {len(sub_categories_two)} for Sub Category ID: {sub_category_id}")
        if not sub_categories_two:
            error_log=f"No sub-categories two found for the given subcategory ID: {sub_category_id}"
            right_to_draw_logs.info(error_log)            
            return Response({"message": "No sub-categories two found for the given subcategory."}, status=404)
         
        serialized_data = SubCategoryTwoSerializer(sub_categories_two, many=True)
        
        return serialized_data
    except Exception as ex:
        error_log=f" Exception Occurred for fetching the Sub Categories Level Two for  {sub_category_id}"
        right_to_draw_logs.info(error_log)
        right_to_draw_logs.error(error_log)
        raise HttpResponseServerError("Exception Occurred for fetching the Sub Categories Level Two.")
    
def get_design_options_for_sub_category(sub_category_id):
    right_to_draw_logs.info(f"Get design options for sub_category_id: {sub_category_id}")
    try:     
        design_options = MstDesignOptions.objects.filter(sub_category_id=sub_category_id)
        right_to_draw_logs.info(f"Design Options: {len(design_options)}")
        
        if not design_options.exists():
            error_log=f"No Design Options found for the given Sub Category ID: {sub_category_id}"
            right_to_draw_logs.info(error_log)            
            raise Http404("No Design Options found for the given Sub Category ID.")
        
        result = []
        for design_option in design_options:
            result.append({
                'design_option_id': design_option.id,
                'desing_option_name': design_option.desing_option_name
            })
        
        right_to_draw_logs.info(f"Number of Design Options: {len(result)} for Sub Category ID: {sub_category_id}")
        return result
    except Http404 as ex:
        raise Http404("Design Options not found for the given Sub Category ID.")
    except Exception as e:
        error_log = f"An error occurred while fetching design options: {str(e)}"
        right_to_draw_logs.info(error_log)
        right_to_draw_logs.error(error_log)
        raise HttpResponseServerError(f"An error occurred while fetching design options: {str(e)}")


def get_design_rules_for_design_option(design_option_ids):
    right_to_draw_logs.info(f"Get design rules for design_option_ids: {design_option_ids}")
    try:
        right_to_draw_logs.info("Prefetching MstDesignOptions and MstDesignOptions__section_groups__rules")
        design_options = MstDesignOptions.objects.prefetch_related('section_groups__rules').filter(id__in=design_option_ids)

        rules_data = []
        unique_rule_ids = set()

        for design_option in design_options:            
            for group in design_option.section_groups.all():                
                for rule in group.rules.all():
                    if rule.id not in unique_rule_ids:
                        unique_rule_ids.add(rule.id)
                        rule_serializer = SectionRulesSerializer(rule)
                        rules_data.append(rule_serializer.data)
                        
        right_to_draw_logs.info(f"Number of Design Rules: {len(rules_data)} for Design Option IDs: {design_option_ids}")
        return rules_data

    except Exception as e:
        error_log = f"An error occurred while fetching design rules: {str(e)}"
        right_to_draw_logs.info(error_log)
        right_to_draw_logs.error(error_log)
        raise Http404(f"An error occurred while fetching design rules: {str(e)}")
    
 
def create_cad_template(data, user):
    component_id = data.get('component')
    component_specifications = data.get('componentSpecifications')
    design_options = data.get('designOptions')
    secondary_sub_level = data.get('secondarySubLevel')
    
    right_to_draw_logs.info(f"Create CAD Template for component_id: {component_id}, component_specifications: {component_specifications}, design_options: {design_options}")
    try:
        component = MstComponent.objects.get(id=component_id)
        right_to_draw_logs.info(f"Component with ID {component_id} found")
    except MstComponent.DoesNotExist:
        error_log=f"Component with ID {component_id} does not exist."
        right_to_draw_logs.info(error_log)
        right_to_draw_logs.error(error_log)
        return None, {"error": "Component not found."}

    data_for_serializer = {
        "opp_number": data.get("oppNumber"),
        "opu_number": data.get("opuNumber"),
        "edu_number": data.get("eduNumber"),
        "model_name": data.get("modelName"),
        "part_number": data.get("partNumber"),
        "revision_number": data.get("revisionNumber"),
        "component_Id": component.pk,
        "pcb_specifications": component_specifications,
        "smt_design_options": design_options,
        "secondary_sub_level": secondary_sub_level,
        'created_by':user.pk,
        'updated_by': user.pk,
        'remarks': data.get('remarks'),
    }

    # Create and validate the serializer
    serializer = CADDesignTemplatesSerializer(data=data_for_serializer)
    log_Str = f"for component_id: {component_id}, Opp Number {data.get('oppNumber')}, \
            Opu Number {data.get('opuNumber')}, Edu Number {data.get('eduNumber')}, \
            Model Name {data.get('modelName')}, Part Number {data.get('partNumber')}, \
            Revision Number {data.get('revisionNumber')}"
    
    if serializer.is_valid():  
        right_to_draw_logs.info(f"CAD Design Template Saving {log_Str}")      
        cad_template = serializer.save()
        return cad_template, None
    else:
        right_to_draw_logs.error(f"Error Saving in CAD Design Template {log_Str}")       
        right_to_draw_logs.info(f"Error Saving in CAD Design Template {log_Str}")       
        return None, serializer.errors


def get_verifier_fields_by_params(component_id=None, category_id=None, sub_category_id=None):    
    right_to_draw_logs.info(f"Get verifier fields by params: component_id: {component_id}, category_id: {category_id}, sub_category_id: {sub_category_id}")
    filter_criteria = Q()
    if component_id:        
        filter_criteria &= Q(component_id=component_id)
    if category_id:        
        filter_criteria &= Q(category_id=category_id)
    if sub_category_id:        
        filter_criteria &= Q(sub_category__id=sub_category_id)
    verifier_fields = MstVerifierField.objects.filter(filter_criteria).distinct().order_by('id')
    right_to_draw_logs.info(f"Verifier Fields: {len(verifier_fields)} for component_id: {component_id}, category_id: {category_id}, sub_category_id: {sub_category_id}")
    serializer = MstVerifierFieldSerializer(verifier_fields, many=True)

    return serializer.data


def create_cad_verifier_template(data, user):
    right_to_draw_logs.info(f"Create CAD Verifier Template")
    
    component_id = data.get('component')
    design_compare_data = data.get('design_compare_data', [])
    verify_compare_data = data.get('verify_compare_data', [])    
    
    try:
        component = MstComponent.objects.get(id=component_id)
        right_to_draw_logs.info(f"Component with ID {component_id} found")
    except MstComponent.DoesNotExist:
        error_log=f"Component with ID {component_id} does not exist."
        right_to_draw_logs.info(error_log)
        right_to_draw_logs.error(error_log)
        return None, {"error": "Component not found."}  
    
    template_data = {
        "opp_number": data.get("oppNumber"),
        "opu_number": data.get("opuNumber"),
        "edu_number": data.get("eduNumber"),
        "model_name": data.get("modelName"),
        "part_number": data.get("partNumber"),
        "revision_number": data.get("revisionNumber"),
        "component_Id": component.pk,
        "pcb_specifications": data.get("componentSpecifications", {}),
        "verifier_query_data": data.get("verifierQueryData", {}),
        'created_by':user.pk,
        'updated_by': user.pk
    }
    log_str = f"for component_id: {component_id}, Opp Number {data.get('oppNumber')}, \
                Opu Number {data.get('opuNumber')}, Edu Number {data.get('eduNumber')}, Model Name {data.get('modelName')}, \
                Part Number {data.get('partNumber')}, Revision Number {data.get('revisionNumber')}"

    # Create and validate the serializer
    serializer = CADVerifierTemplateSerializer(data=template_data)
    if serializer.is_valid():
        right_to_draw_logs.info(f"CAD Verifier Template Saving {log_str}")
        cad_verifier_template = serializer.save()            
        return cad_verifier_template, None
    else:
        right_to_draw_logs.error(f"Error Saving in CAD Verifier Template {log_str} -- Error is {serializer.errors}")
        right_to_draw_logs.info(f"Error Saving in CAD Verifier Template {log_str} -- Error is {serializer.errors}")        
        return None, serializer.errors


def condition_operators(condition, target_value):
    is_comparision_data_available = False    
    is_deviated = False
    target_value = float(target_value)
    if condition.comparison_operator == 'range':
        if (condition.comparison_min_value and condition.comparison_max_value):
            is_comparision_data_available = True
            if not (target_value >= condition.comparison_min_value  and \
                    target_value <= condition.comparison_max_value):
                is_deviated = True        

    elif condition.comparison_operator == 'eq':
        if condition.comparison_min_value:
            is_comparision_data_available = True
            if not target_value == condition.comparison_min_value:
                is_deviated = True
    elif condition.comparison_operator == 'gte':
        if condition.comparison_max_value:
            is_comparision_data_available = True
            if not target_value >= condition.comparison_max_value:
                is_deviated = True
    elif condition.comparison_operator == 'lte':
        if condition.comparison_min_value:
            is_comparision_data_available = True
            if not target_value <= condition.comparison_min_value:
                is_deviated = True
    elif condition.comparison_operator == 'gt':
        if condition.comparison_max_value:
            is_comparision_data_available = True
            if not target_value > condition.comparison_max_value:
                is_deviated = True
    elif condition.comparison_operator == 'lt':
        if condition.comparison_min_value:
            is_comparision_data_available = True    
            if not target_value < condition.comparison_min_value:
                is_deviated = True
    return is_deviated, is_comparision_data_available


def check_conditions(sub_category, category, pcb_specifications_inp):
    res = []
    pcb_specifications = {int(k):id(v) for k, v in pcb_specifications_inp.items()}    
    conditions = MstConditions.objects.filter(Q(category=category.pk) | Q(subcategory=sub_category.pk))  
    right_to_draw_logs.info(f"Checking conditions for sub_category: {sub_category.name}, conditions: {len(conditions)}")
    
    # Group conditions by condition_variable and comparison_variable
    grouped_conditions = defaultdict(list)
    for condition in conditions:
        grouped_conditions[(condition.condition_variable, condition.comparison_variable)].append(condition)
    
    is_prequisite_meet = False
    
    for (condition_variable, comparison_variable), group in grouped_conditions.items():
        right_to_draw_logs.info(f"Checking group for condition_variable: {condition_variable}, compare_variable: {comparison_variable}")
        
        # Flag to track if pre-requisite is met for this group
        group_prequisite_met = False
        group_deviation_found = False
        atleast_one_conditon_sat_compare = False      

        condition_variable = condition_variable.strip().lower()
        comparison_variable = comparison_variable.strip().lower()
        
        try:
            condition_id = MstCategory.objects.get(category_name__iexact=condition_variable).id
            comparision_id = MstCategory.objects.get(category_name__iexact=comparison_variable).id
        except ObjectDoesNotExist as ex:
            right_to_draw_logs.info(f"Invalid Category Name submitted: {condition_variable} or {comparison_variable} for Condition Check")
            right_to_draw_logs.info(f"Exception Occurred: {ex}")
            deviation_result = {
                'categor_id' : category.pk,
                'name': category.category_name,
                'selected_deviation_id': "N/A",
                'selected_deviation_name': "N/A",
                'is_deviated': True
            }
            res.append(deviation_result)
            continue

        selected_val = pcb_specifications.get(condition_id)
        target_val = pcb_specifications.get(comparision_id)            

        # For B14 Size and Dielectric Thickness we are having the Text Boxes so we need to convert the values to float
        # For B11 and B17 the values are selected from drop down, so in the given data we are going to have the ids
        # of the drop down values, so we need to fetch from the the MstSubCategory table and get the name and then convert
        # to float
        if condition_variable.strip().lower() == 'b14 size':
            selected_val = float(selected_val)
        elif comparison_variable.strip().lower() == 'dielectric thickness':
            target_val = float(target_val)
        else:
            selected_val = MstSubCategory.objects.get(id=selected_val).name
            target_val = MstSubCategory.objects.get(id=target_val).name
            selected_val = float(selected_val)
            target_val = float(target_val)            

        if not selected_val:
            right_to_draw_logs.info(f"{condition_variable} Value does not exists in the input for Condition Check")
            deviation_result = {
                'categor_id' : condition_id,
                'name': condition_variable,
                'selected_deviation_id': "N/A",
                'selected_deviation_name': "N/A",
                'is_deviated': True
            }
            res.append(deviation_result)

            if not target_val:
                right_to_draw_logs.info(f"{comparison_variable} Value does not exists in the input for Condition Check")
                deviation_result = {
                    'categor_id' : comparision_id,
                    'name': comparison_variable,
                    'selected_deviation_id': "N/A",
                    'selected_deviation_name': "N/A",
                    'is_deviated': True
                }
                res.append(deviation_result)

            continue

        for condition in group:
            right_to_draw_logs.info(f"Processing condition: {condition.condition_variable} & {condition.comparison_variable}")                                    
            selected_val = float(selected_val)            
            is_condition_satisfied = False            
            
            right_to_draw_logs.info(f"Selected Value is : {selected_val} and Target Value is {target_val}")                    
            if condition.condition_operator == 'range':
                if selected_val >= condition.condition_min_value and selected_val < condition.condition_max_value:                    
                    is_condition_satisfied = True
            elif condition.condition_operator == 'gte':      
                if selected_val >= condition.condition_max_value:
                    is_condition_satisfied = True
            elif condition.condition_operator == 'lte':
                if selected_val <= condition.condition_min_value:
                    is_condition_satisfied = True
            elif condition.condition_operator == 'gt':
                if selected_val > condition.condition_max_value:
                    is_condition_satisfied = True
            elif condition.condition_operator == 'lt':
                if selected_val < condition.condition_min_value:
                    is_condition_satisfied = True
            elif condition.condition_operator == 'eq':
                if selected_val == condition.condition_min_value:
                    is_condition_satisfied = True
            # For Category B11 and B17 we are having the conditions like Add and Sub So we don't need to check the 
            # Condition Operator Range values so Selected Val plus or Minus of the selected target value is fine
            elif condition.condition_operator == 'add':
                group_prequisite_met = True
                if selected_val + condition.condition_min_value != target_val:                                                        
                    group_deviation_found = True
            elif condition.condition_operator == 'sub':
                group_prequisite_met = True
                if selected_val - condition.condition_min_value != target_val:                                                        
                    group_deviation_found = True
            else:
                continue
            
            right_to_draw_logs.info(f"Is Basic Condition Satisfied {is_condition_satisfied}")
            # As of now no need to execute this condition for add or sub operators it will be use for 
            # comparision operators Even If it executes nothing will happen, since we do not have
            # conditon operator for this
            if is_condition_satisfied:
                group_prequisite_met = True
                group_deviation_found, comparision_data_availablity = condition_operators(condition, target_val)
                if comparision_data_availablity:
                    atleast_one_conditon_sat_compare = True
                
            if group_deviation_found:
                break
        
        if not group_prequisite_met:
            right_to_draw_logs.info(f"Pre-requisite not met for {condition_variable} with comparison {comparison_variable}.")            
            deviation_result = {
                'categor_id' : condition_id,
                'name': condition_variable,
                'selected_deviation_id': selected_val,
                'selected_deviation_name': selected_val,
                'is_deviated': True
            }
            res.append(deviation_result)

        if group_prequisite_met and group_deviation_found:
            is_deviated = True
            right_to_draw_logs.info(f"{condition_variable} Value does not meet criteria of the conditons mentioned")
            deviation_result = {
                'categor_id' : comparision_id,
                'name': comparison_variable,
                'selected_deviation_id': target_val,
                'selected_deviation_name': target_val,
                'is_deviated': True
            }
            res.append(deviation_result)

            right_to_draw_logs.info(f"Condition variable {condition_variable} is deviated.")

    return res


def append_design_response_to_final_response(design_verification_res, condition):
    existing_record = next((item for item in design_verification_res if item["categor_id"] == condition["categor_id"]), None)                                        
    if existing_record:
        # If the condition's is_deviated is True, update it
        if condition["is_deviated"]:
            existing_record["is_deviated"] = condition["is_deviated"]
    else:
        # If no existing record, append the new condition
        design_verification_res.append(condition)
    return design_verification_res

                        
def compare_verifier_data_with_design_data(data):        
    opp_number = data.get("oppNumber")
    opu_number = data.get("opuNumber")
    edu_number = data.get("eduNumber")
    model_name = data.get("modelName")
    part_number = data.get("partNumber")
    revision_number = data.get("revisionNumber")
    component_id = data.get('component')
    
    log_str = f"for component_id: {component_id}, Opp Number {opp_number}, \
                Opu Number {opu_number}, Edu Number {edu_number}, Model Name {model_name}, \
                Part Number {part_number}, Revision Number {revision_number}"
    
    right_to_draw_logs.info(f"The Verification Rules Started {log_str}")
                
    input_specifications_data = data.get('componentSpecifications')

    design_verification_res = []

    template = CADDesignTemplates.objects.filter(
        opp_number=opp_number,
        opu_number=opu_number,
        edu_number=edu_number,
        model_name=model_name,
        part_number=part_number,
        revision_number=revision_number,
        component_Id=component_id
    ).first()

    if not template:
        right_to_draw_logs.error(f"No matching CADDesignTemplate found for Verifier Template {log_str}")
        return {}       
    
    pcb_specifications_d = template.pcb_specifications 
    design_template_specification_data = {int(k):int(v) for k, v in pcb_specifications_d.items()}
    
    component = MstComponent.objects.get(id=component_id)
    # Looping Each record from the verfier template
    # Here B14 Size and Dielectric Thickness is going to be the Text Boxes    
    for category_id, selected_sub_category_id in input_specifications_data.items():
        is_deviated = False
        try:
            category_id = int(category_id)        
            try:
                category = MstCategory.objects.get(id=category_id)
            except ObjectDoesNotExist as ex:
                right_to_draw_logs.info(f"Design Verifications - Invalid Category Id submitted: {category_id} for component_id: {component_id}")
                is_deviated = True
                deviation_result = {
                    'categor_id' : category_id,
                    'name': "Invalid Category Id",
                    'selected_deviation_id': "N/A",
                    'selected_deviation_name': "N/A",
                    'is_deviated': True
                }
                design_verification_res = append_design_response_to_final_response(design_verification_res, deviation_result)            
                continue
            
            right_to_draw_logs.info(f"Design Verifiation started for : {category.category_name.strip()} -- Value Selected {selected_sub_category_id}")   
            
            # For Dielectric Thickness, The Verifer enter the value manually Entered Value Instead 
            # of selecting from Drop Down
            if (category.category_name.strip() == 'Dielectric Thickness' or 
                category.category_name.strip() == 'B14 Size') and \
                component.component_name.strip().lower() == 'b14':

                right_to_draw_logs.info(f"Validating the {category.category_name} & category value {selected_sub_category_id}")
                selected_val = float(selected_sub_category_id)
                
                # Value should match with the value selected in Design Template
                # Steps to Check:
                # Get the Sub-Category Id of selected in Design Template
                # Get the Value of the Sub-Category from Name column
                # Compare the Both the values Should be match
                if selected_val == 0:                    
                    deviation_result = {
                        'categor_id' : category_id,
                        'name': category.category_name,
                        'selected_deviation_id': "N/A",
                        'selected_deviation_name': selected_val,
                        'is_deviated': False
                    }
                    design_verification_res = append_design_response_to_final_response(design_verification_res, deviation_result)            
                    continue
                
                if category_id in design_template_specification_data:
                    right_to_draw_logs.info(f"{category.category_name} Present in the Design Template {category_id}")
                    # Get the selected value during the Design template                
                    try:
                        sub_category_id = design_template_specification_data.get(category_id)
                        sub_category = MstSubCategory.objects.get(id=sub_category_id)
                    except ObjectDoesNotExist:
                        deviation_result = {
                            'categor_id' : category_id,
                            'name': category.category_name,
                            'selected_deviation_id': "Not Availble",
                            'selected_deviation_name': "Not Availble",
                            'is_deviated': True
                        }                        
                        design_verification_res = append_design_response_to_final_response(design_verification_res, deviation_result)            
                        continue

                    if category.category_name.strip() == 'Dielectric Thickness': 
                        design_val = float(sub_category.name.strip('"'))

                        if design_val != selected_val:
                            is_deviated = True
                    elif category.category_name.strip() == 'B14 Size':
                        if 'less than or equal to 0.800' in sub_category.name.strip().lower():
                            if selected_val > 0.800:
                                is_deviated = True
                        elif 'greater than 0.800' in sub_category.name.strip().lower():
                            if selected_val <= 0.800:
                                is_deviated = True

                    deviation_result = {
                        'categor_id' : category_id,
                        'name': category.category_name,
                        'selected_deviation_id': "N/A",
                        'selected_deviation_name': selected_val,
                        'is_deviated': is_deviated
                    }                               
                    design_verification_res = append_design_response_to_final_response(design_verification_res, deviation_result)
                    right_to_draw_logs.info(f"The Dielectric Thickness result for {selected_sub_category_id}, component_id: {component_id} is {is_deviated}")
                    continue
            

            else:
                try:
                    sub_category = MstSubCategory.objects.get(id=int(selected_sub_category_id))
                except ObjectDoesNotExist:
                    right_to_draw_logs.info(f"Design Verifications - Invalid Sub Category Id submitted: {selected_sub_category_id} for Category Id{category_id} & component_id: {component_id}")                
                    deviation_result = {
                        'categor_id' : category_id,
                        'name': category.category_name,
                        'selected_deviation_id': selected_sub_category_id,
                        'selected_deviation_name': "N/A",
                        'is_deviated': True
                    }
                    design_verification_res = append_design_response_to_final_response(design_verification_res, deviation_result)
                    continue                
                if category_id in design_template_specification_data:            
                    if design_template_specification_data.get(category_id) != int(selected_sub_category_id):
                        is_deviated = True                
                    else:
                        is_deviated = False                    
                else:
                    is_deviated = False
                
                right_to_draw_logs.info(f"Design Verifications - Category Id: {category_id}, is_deviated: {is_deviated}")
                deviation_result = {
                    'categor_id' : category_id,
                    'name': category.category_name,
                    'selected_deviation_id': selected_sub_category_id,
                    'selected_deviation_name': sub_category.sub_category_name,
                    'is_deviated': is_deviated 
                }
                design_verification_res = append_design_response_to_final_response(design_verification_res, deviation_result)
                
                if MstConditions.objects.filter(Q(category=category_id) | Q(subcategory=sub_category.pk)).exists():
                    conditons_res = check_conditions(sub_category, category, input_specifications_data)
                    # Iterate over the conditions response
                    for condition in conditons_res:
                        design_verification_res = append_design_response_to_final_response(design_verification_res, condition)
                
        except Exception as ex:
            right_to_draw_logs.info(f"Exception Occurred for {category_id} :{selected_sub_category_id} -- {ex}")        
    
    right_to_draw_logs.info(f"Design Verification completed for {log_str}")
    return design_verification_res


def get_dielectric_material_selected_value(data):
    die_material_val = None
    for category_id, selected_sub_category_id in data.items():
        try:
            category_id = int(category_id)        
            try:
                category = MstCategory.objects.get(id=category_id)
            except ObjectDoesNotExist as ex:
                continue
            if category.category_name.strip().lower() == 'dielectric material':
                sub_category = MstSubCategory.objects.get(id=selected_sub_category_id)
                die_material_val = sub_category.name
                break
        except Exception as ex:
            right_to_draw_logs.info("Unable to fetch the Dielectric Material Value")
    return die_material_val


def comapre_verfier_data_with_rules(verifier_id, field_value, design_data, ceramic_reasonator_value):    
    try:
        right_to_draw_logs.info(f"Compare verifier data with rules for verifier_id: {verifier_id}, field_value: {field_value}") 
        verifier_field = MstVerifierField.objects.get(id=int(verifier_id))

        verifier_rule = []
        if ceramic_reasonator_value:
            filtered_rules = MstVerifierRules.objects.filter(verifier_field=verifier_field.pk)
            for rule in filtered_rules:
                if rule.conditional_var.strip().lower() == 'ceramic_resonator_size':
                    rul_val = rule.value.strip().lower()
                    if rul_val and float(rul_val) == float(ceramic_reasonator_value):
                        verifier_rule.append(rule)
                # Ceramic Reasonator is for few rules only applicable, So for other rules
                # the value is empty so we can go ahead and add if the value is empty
                elif rule.conditional_var is None or rule.conditional_var == '':
                    verifier_rule.append(rule)
        else:
            verifier_rule = list(MstVerifierRules.objects.filter(verifier_field=verifier_field.pk))            

        if not verifier_rule:
            right_to_draw_logs.info(f"No verifier rule found for verifier_id {verifier_id}")
            return False
        else:
            verifier_rule = verifier_rule[0]
        
        right_to_draw_logs.info(f"Verifier Rule found for verifier_id {verifier_id} --- is : {verifier_rule.rule_number}-{verifier_rule.design_doc}")

        rules = verifier_rule.rule_number
        rule_numbers = rules.split(',')
        is_deviation = False
        for rule_number in rule_numbers:
            rule_number = rule_number.strip()
            if verifier_field.field_name.strip() == 'Enter minimum dot size for mounting of Transformers in a DWG':
                die_material_val =  get_dielectric_material_selected_value(design_data)
                # For 'Mounting of Transformers' If Dielectric Material is 'Rogers: RO4350B'
                # Then only '4.11.2' rule needs to execute else Do not execute
                # Vice Versa.
                if die_material_val == 'Rogers: RO4350B' and rule_number != '4.11.2':
                    right_to_draw_logs.info(f"Dielectric Material is 'Rogers: RO4350B' and rule number is not '4.11.2', So Skipping the Rule")
                    continue                
                if die_material_val != 'Rogers: RO4350B' and rule_number == '4.11.2':
                    right_to_draw_logs.info(f"Dielectric Material is not 'Rogers: RO4350B' and rule number is '4.11.2', So Skipping the Rule")
                    continue            
            design_doc = verifier_rule.design_doc
            design_doc = design_doc.strip()
            section_rule = MstSectionRules.objects.get(rule_number=rule_number, design_doc=design_doc)            
            try:        
                min_value = float(section_rule.min_value) if section_rule.min_value else None                   
            except Exception as e:
                min_value = None


            try:                    
                max_value = float(section_rule.max_value) if section_rule.max_value else None        
            except Exception as e:
                max_value = None
            if (min_value is not None and field_value < min_value) or (max_value is not None and field_value > max_value):
                is_deviation = True
                break
        return is_deviation

    except ObjectDoesNotExist as ex:
        right_to_draw_logs.info(f"Verifier Field or Rule not found for verifier id {verifier_id}: {str(ex)}")
        right_to_draw_logs.error(f"Verifier Field or Rule not found for verifier id {verifier_id}: {str(ex)}")
        return False
    except Exception as ex:
        right_to_draw_logs.error(f"An error occurred while comparing verifier data with rules for verifier id {verifier_id}: {str(ex)}")
        right_to_draw_logs.info(f"An error occurred while comparing verifier data with rules for verifier id {verifier_id}: {str(ex)}")
        return False


def get_category_for_b11_value(design_data):
    category_b11_val = None
    for category_id, selected_sub_category_id in design_data.items():
        try:
            category_id = int(category_id)        
            try:
                category = MstCategory.objects.get(id=category_id)
            except ObjectDoesNotExist as ex:
                continue
            if category.category_name.strip().lower() == 'category for b11':
                sub_category = MstSubCategory.objects.get(id=selected_sub_category_id)
                category_b11_val = sub_category.name
                break
        except Exception as ex:
            right_to_draw_logs.info("Unable to fetch the Dielectric Material Value")
    return category_b11_val


def get_ceramic_reasonator_selected_value(design_data):
    ceramic_reasonator_val = None
    for design_id, sub_category_id in design_data.items():
        try:
            design_id = int(design_id)        
            try:
                category_rec = MstCategory.objects.get(id=design_id)
            except ObjectDoesNotExist as ex:
                continue
            category_name = category_rec.category_name.strip().strip('.').strip().lower()
            if category_name == 'ceramic resonator size requirement for b11':
                sub_category_rec = MstSubCategory.objects.get(id=sub_category_id)
                ceramic_reasonator_val = sub_category_rec.sub_category_name.strip().lower()
                ceramic_reasonator_val = int(ceramic_reasonator_val)
                break
        except Exception as ex:
            right_to_draw_logs.info("Unable to fetch the Ceramic Resonator Value")
    return ceramic_reasonator_val

def comapre_verfier_data(verified_data, design_data, component_id):
    verifier_res = []
    ceramic_reasonator_value = None
    component = MstComponent.objects.get(id=component_id)  
    if component.component_name.strip().lower() == 'b11':
        category_b11_val = get_category_for_b11_value(design_data)
        if category_b11_val.strip().lower() == 'ceramic resonator technology':
            ceramic_reasonator_value = get_ceramic_reasonator_selected_value(design_data)

    for id, val in verified_data.items():
        val = float(val)
        try:
            verifier_field = MstVerifierField.objects.get(id=id)
            name = verifier_field.name
        except ObjectDoesNotExist as ex:
            right_to_draw_logs.info(f"Verifier Field not found for verifier id {id}: {str(ex)}")
            right_to_draw_logs.error(f"Verifier Field not found for verifier id {id}: {str(ex)}")
            data = {'id' :id, 'name':'Not Applicable', 'value':val, 'is_deviated':True}
        if val == 0:
            data = {'id' :id, 'name':name, 'value':val, 'is_deviated':False}
        else:
            is_deviated = comapre_verfier_data_with_rules(id, val, design_data, ceramic_reasonator_value)
            data = {'id' :id, 'name':name, 'value':val, 'is_deviated':is_deviated}
        verifier_res.append(data)
    return verifier_res


def compare_verifier_data_with_rules_and_designs(data):    
    res = {
        "opp_number": data.get("oppNumber"),
        "opu_number": data.get("opuNumber"),
        "edu_number": data.get("eduNumber"),
        "model_name": data.get("modelName"),
        "part_number": data.get("partNumber"),
        "revision_number": data.get("revisionNumber"), 
        "component_id": data.get('component')     
    }

    result_string = ", ".join(f"{key}: {value}" for key, value in res.items())
    
    design_specification_data = compare_verifier_data_with_design_data(data)
    if not design_specification_data:
        return {}
    res['verify_design_fields_data']= design_specification_data

    right_to_draw_logs.info(f"Compare verifier data with rules and designs for {result_string}")    
    verified_rule_data = comapre_verfier_data(data.get("verifierQueryData"), data.get('componentSpecifications'),  data.get('component'))
    res['verified_query_data'] = verified_rule_data


    right_to_draw_logs.info(f"Verification Completed for {result_string}")
    return res


def get_verifier_record(request_data):
    """
    This function takes the request data, queries the CADVerifierTemplates table
    based on the parameters provided, and returns the data in the required format.
    """
    right_to_draw_logs.info(f"Get verifier record for request data")
    # Extract the parameters from the request data
    right_to_draw_logs.info("Extract the parameters from the request data")
    opp_number = request_data.get('oppNumber')
    opu_number = request_data.get('opuNumber')
    edu_number = request_data.get('eduNumber')
    model_name = request_data.get('modelName')
    part_number = request_data.get('partNumber')
    revision_number = request_data.get('revisionNumber')
    component_id = request_data.get('component')

    # Query the CADVerifierTemplates table based on the parameters
    try:
        right_to_draw_logs.info(f"Querying CADVerifierTemplates table to get verifier record for parameters: {opp_number}, {opu_number}, {edu_number}, {model_name}, {part_number}, {revision_number}, {component_id}")
        verifier_record = CADVerifierTemplates.objects.get(
            opp_number=opp_number,
            opu_number=opu_number,
            edu_number=edu_number,
            model_name=model_name,
            part_number=part_number,
            revision_number=revision_number,
            component_Id=component_id
        )
    except ObjectDoesNotExist as ex:
        error_log = f"Verifier record not found: {str(ex)}"
        right_to_draw_logs.info(error_log)
        right_to_draw_logs.error(error_log)
        raise ObjectDoesNotExist('Verifier record not found.')

    # Prepare the response data in the required format
    response_data = {
        'oppNumber': verifier_record.opp_number,
        'opuNumber': verifier_record.opu_number,
        'eduNumber': verifier_record.edu_number,
        'modelName': verifier_record.model_name,
        'partNumber': verifier_record.part_number,
        'revisionNumber': verifier_record.revision_number,
        'component': verifier_record.component_Id.id,

        'componentSpecifications': verifier_record.pcb_specifications,
        'verifierQueryData': verifier_record.verifier_query_data
    }
    right_to_draw_logs.info(f"Response data prepared for the queried verifier record")
    return response_data


def save_approver_results(data, user):
    try:
        template_data = {
            "opp_number": data.get("oppNumber"),
            "opu_number": data.get("opuNumber"),
            "edu_number": data.get("eduNumber"),
            "model_name": data.get("modelName"),
            "part_number": data.get("partNumber"),
            "revision_number": data.get("revisionNumber"),
            "component_Id":data.get('component'),
            "pcb_specifications": data.get("componentSpecifications", {}),
            "approver_data": data.get("approverQueryData", {}),
            "status": data.get("status"),
            "comments": data.get("comments"),
            "created_by": user.id,
            "updated_by": user.id
        }
        serializer = CADApproverTemplateSerializer(data=template_data)

        log_str = f"for component_id: {data.get('component')}, Opp Number {data.get('oppNumber')}, \
                    Opu Number {data.get('opuNumber')}, Edu Number {data.get('eduNumber')}, Model Name {data.get('modelName')}, \
                    Part Number {data.get('partNumber')}, Revision Number {data.get('revisionNumber')}"
        
        if serializer.is_valid():        
            template = serializer.save()
            return template, None
            
        else:
            right_to_draw_logs.error(f"Error Saving in Approver Template {log_str} -- Error is {serializer.errors}")
            right_to_draw_logs.info(f"Error Saving in Approver Template {log_str} -- Error is {serializer.errors}")        
            return None, serializer.errors
    except Exception as ex:
        right_to_draw_logs.error(f"An error occurred while saving approver template: {str(ex)}")
        right_to_draw_logs.info(f"An error occurred while saving approver template: {str(ex)}")
        return None, str(ex)