from django.urls import path

from .views import ComponentAPIView, SubCategoryTwoAPIView,\
    CADDesignTemplatesAPIView, DesignOptionAPIView,DesignRuleAPIView, \
        CADVerifierTemplateCreateAPIView, MstVerifierFieldFilterAPIView, \
        MstVerifierFieldResultAPIView, ApproverAPIView, CheckDesignerAndVerifierRecordAPIView, UserCreatedTemplatesView

urlpatterns = [
    # Pull ALl the categories and Sub Categories for the Given Component
    path('pcb-specification/<int:component_id>/', ComponentAPIView.as_view()),    
    # Pull Level 2 Drop down values for a given category Id
    path('sub-categories-two/<int:sub_category_id>/', SubCategoryTwoAPIView.as_view(), name='sub-categories-two'),
    # Get /Post the Cad Design Template for the submitting User
    path('cad-design-templates/<int:id>/', CADDesignTemplatesAPIView.as_view(), name='design-templates'),
    path('cad-design-templates/', CADDesignTemplatesAPIView.as_view(), name='design-templates'),
    # Get Design Options for the specific Sub Category selected.
    path('design-options/<int:sub_category_id>/', DesignOptionAPIView.as_view(), name='design-options'),
    # Get Design Rules for specified designs , Pass design option Ids in Query Param
    path('design-rules/', DesignRuleAPIView.as_view(), name='design-rules'),
    # Get All the Verifier field records for given filter params    
    path('verifier-fields/', MstVerifierFieldFilterAPIView.as_view(), name='verifier-fields'),
    # Save the Verifier Templates
    path('verifier-templates/', CADVerifierTemplateCreateAPIView.as_view(), name='verifier-templates'),
    # Verify result screen 
    path('verify-results/', MstVerifierFieldResultAPIView.as_view(), name='verify-results'),    
    # Post the Approver Template
    path('approver-template/',ApproverAPIView.as_view(),name='approver-templates'),
    # Check for the Verfier and Approver paths     
    path('check-template/', CheckDesignerAndVerifierRecordAPIView.as_view(), name='check-apprvoer-record'),
    path('user-templates/', UserCreatedTemplatesView.as_view(), name='user-templates'),
]