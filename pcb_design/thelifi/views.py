"""
thelifi.views

API endpoints for:
- Uploading filter files (S2P, simulation) and creating a submission record.
- Calculating KPIs and generating summary Excel files.
- Generating plots based on submission and plot configuration.
- Listing all files in a submission folder.

All endpoints include error handling, logging, and OpenAPI/Swagger documentation.
"""

import time
import os
import io
import uuid
import glob
import shutil
import warnings
import numpy as np
import pandas as pd
import skrf as rf

from django.conf import settings
from django.http import FileResponse

from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny

from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from . import thelifi_logs
from .models import FilterSubmission
from .plotGeneration import (
    generate_s_parameter_plots_only,
    generate_statistical_and_histogram_plots_only
)
from .serializers import FilterSubmissionSerializer

class FilterUploadView(APIView):
    """
    Upload model details with multiple S2P and Simulation files.
    Creates a submission record and stores files in a unique folder.
    """
    parser_classes = [MultiPartParser, FormParser]

    @swagger_auto_schema(
        operation_description="Upload model details with multiple S2P and Simulation files.",
        manual_parameters=[
            openapi.Parameter('model_number', openapi.IN_FORM, description="Model Number", type=openapi.TYPE_STRING, required=True),
            openapi.Parameter('edu_number', openapi.IN_FORM, description="EDU Number", type=openapi.TYPE_STRING, required=True),
            openapi.Parameter('filter_type', openapi.IN_FORM, description="Filter Type", type=openapi.TYPE_STRING, required=True),
            openapi.Parameter('s2p_files', openapi.IN_FORM, description="S2P Files (multiple files, use same key multiple times)", type=openapi.TYPE_FILE, required=True),
            openapi.Parameter('simulation_files', openapi.IN_FORM, description="Simulation Files (multiple files, use same key multiple times)", type=openapi.TYPE_FILE, required=False),
        ],
        responses={201: "Files uploaded successfully", 400: "Bad Request"}
    )
    def post(self, request):
        """
        Handle file upload and submission creation.
        """
        try:
            model_number = request.data.get('model_number')
            edu_number = request.data.get('edu_number')
            filter_type = request.data.get('filter_type')
            s2p_files = request.FILES.getlist('s2p_files')
            simulation_files = request.FILES.getlist('simulation_files')

            if not (model_number and edu_number and filter_type and s2p_files):
                thelifi_logs.warning("Upload failed: Missing required fields.")
                return Response({"error": "All required fields must be provided."}, status=status.HTTP_400_BAD_REQUEST)

            folder_name = f"{model_number}_{edu_number}_{uuid.uuid4().hex[:6]}"
            base_path = os.path.join(settings.MEDIA_ROOT, 'uploads', folder_name)
            s2p_path = os.path.join(base_path, 's2pFiles')
            sim_path = os.path.join(base_path, 'Simulation_Files')
            plot_path = os.path.join(base_path, 'Generated_Plots')

            os.makedirs(s2p_path, exist_ok=True)
            os.makedirs(sim_path, exist_ok=True)
            os.makedirs(plot_path, exist_ok=True)

            for file in s2p_files:
                file_path = os.path.join(s2p_path, file.name)
                with open(file_path, 'wb+') as destination:
                    for chunk in file.chunks():
                        destination.write(chunk)

            for file in simulation_files:
                file_path = os.path.join(sim_path, file.name)
                with open(file_path, 'wb+') as destination:
                    for chunk in file.chunks():
                        destination.write(chunk)

            submission = FilterSubmission.objects.create(
                folder_name=folder_name,
                model_number=model_number,
                edu_number=edu_number,
                filter_type=filter_type
            )

            thelifi_logs.info(f"Files uploaded for submission_id={submission.id}, folder={folder_name}")
            return Response({"message": "Files uploaded successfully.", "submission_id": submission.id, "folder_name": folder_name}, status=status.HTTP_201_CREATED)
        except Exception as e:
            thelifi_logs.error(f"Error in FilterUploadView: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class KPI_CalculationAPIView(APIView):
    """
    Calculate KPIs and generate summary Excel for a submission.
    """
    @swagger_auto_schema(
        operation_description="Calculate KPIs and generate summary Excel.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            required=['submission_id', 'kpi_data'],
            properties={
                'submission_id': openapi.Schema(type=openapi.TYPE_INTEGER, description='ID of the uploaded submission record'),
                'kpi_data': openapi.Schema(type=openapi.TYPE_OBJECT, description='KPI JSON configuration'),
            }
        ),
        responses={200: "Summary generated successfully", 400: "Bad Request", 404: "Submission not found"}
    )
    def post(self, request):
        """
        Calculate KPIs and generate summary Excel.
        """
        submission_id = request.data.get('submission_id')
        kpi_data = request.data.get('kpi_data')

        if not submission_id or not kpi_data:
            thelifi_logs.warning("KPI calculation failed: Missing submission_id or kpi_data.")
            return Response({"error": "submission_id and kpi_data are required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            submission = FilterSubmission.objects.get(id=submission_id)
        except FilterSubmission.DoesNotExist:
            thelifi_logs.warning(f"KPI calculation: Submission not found for id={submission_id}")
            return Response({"error": "Submission not found."}, status=status.HTTP_404_NOT_FOUND)

        base_path = os.path.join(settings.MEDIA_ROOT, 'uploads', submission.folder_name)
        s2p_path = os.path.join(base_path, 's2pFiles')
        plot_path = os.path.join(base_path, 'Generated_Plots')

        try:
            excel_path, summary_json, per_file_json = self.generate_summary(s2p_path, plot_path, kpi_data, base_path)
            thelifi_logs.info(f"KPI summary generated for submission_id={submission_id}")
            return Response({
                "message": "Summary generated successfully.",
                "excel_file_path": excel_path.replace(settings.MEDIA_ROOT, '/media'),
                "kpi_config_data": kpi_data,
                "summary_data": summary_json,
                "per_file_data": per_file_json,
            }, status=status.HTTP_200_OK)
        except Exception as e:
            thelifi_logs.error(f"Error in KPI_CalculationAPIView: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def generate_summary(self, s2p_folder, plot_folder, kpi_config, base_path):
        """
        Generate summary Excel file for KPIs.
        """
        records = []
        s2p_files = glob.glob(os.path.join(s2p_folder, 'U*.s2p'))

        if not s2p_files:
            thelifi_logs.error("No U*.s2p files found in the provided folder.")
            raise FileNotFoundError("No U*.s2p files found in the provided folder.")

        def convert_type2(path):
            with open(path) as fh:
                lines = fh.readlines()
            header = [l for l in lines if l.lstrip().startswith(("!", "#"))]
            data = [l.strip() for l in lines if not l.lstrip().startswith(("!", "#")) and l.strip()]
            if not (len(data) >= 3 and len(data[0].split()) == 1 and len(data[1].split()) == 4):
                return path
            merged = [data[i] + " " + data[i + 1] + " " + data[i + 2] + "\n" for i in range(0, len(data) - 2, 3)]
            tmp = path + "__fixed.s2p"
            with open(tmp, "w") as out:
                for h in header:
                    out.write(h if h.endswith("\n") else h + "\n")
                out.writelines(merged)
            return tmp

        def kpis_for_network(ntwk, CFG):
            f = ntwk.f
            s11, s21, s22 = ntwk.s[:, 0, 0], ntwk.s[:, 1, 0], ntwk.s[:, 1, 1]
            mag21 = 20 * np.log10(np.abs(s21))
            mag11 = 20 * np.log10(np.abs(s11))
            mag22 = 20 * np.log10(np.abs(s22))
            phase = np.unwrap(np.angle(s21))
            gd_ns = np.round(-np.gradient(phase, f) / (2 * np.pi) * 1e9, 2)

            out = []
            for kpi, bands in CFG["KPIs"].items():
                for b in bands:
                    lo, hi = b["range"]
                    mask = (f >= lo) & (f <= hi)

                    if kpi == "IL":
                        val = round(-mag21[mask].min(),2)
                    elif kpi == "RL":
                        val = round(min(-mag11[mask].max(), -mag22[mask].max()),2)
                    elif kpi == "Flat":
                        val = round(mag21[mask].max() - mag21[mask].min(),2)
                    elif kpi == "GD":
                        val = round(gd_ns[mask].max(),2)
                    elif kpi == "GDV":
                        val = round(np.ptp(gd_ns[mask]),2)
                    elif kpi == "LPD_MIN":  
                        A = np.vstack([f[mask], np.ones_like(f[mask])]).T  
                        slope, intercept = np.linalg.lstsq(A, phase[mask], rcond=None)[0]  
                        resid = phase - (slope * f + intercept)  
                        lpd_deg = np.degrees(resid)  
                        val = round(lpd_deg[mask].min(),2)
                    elif kpi == "LPD_MAX":  
                        A = np.vstack([f[mask], np.ones_like(f[mask])]).T  
                        slope, intercept = np.linalg.lstsq(A, phase[mask], rcond=None)[0]  
                        resid = phase - (slope * f + intercept)  
                        lpd_deg = np.degrees(resid)  
                        val = round(lpd_deg[mask].max(),2) 
                    else:
                        val = np.nan
                    out.append(val)

            for sb in CFG["StopBands"]:
                lo, hi = sb["range"]
                rej = round(-mag21[(f >= lo) & (f <= hi)].max(),2)
                out.append(rej)
            return out

        for p in s2p_files:
            cp = convert_type2(p)
            try:
                nt = rf.Network(cp)
            except Exception as e:
                warnings.warn("Skipping " + p + ": " + str(e))
                thelifi_logs.warning(f"Skipping file {p}: {str(e)}")
                continue
            records.append([os.path.basename(p)] + kpis_for_network(nt, kpi_config))

        cols = ["File"]
        for kpi, bands in kpi_config["KPIs"].items():
            cols.extend([kpi + "_" + b["name"] for b in bands])
        cols.extend([sb["name"] for sb in kpi_config["StopBands"]])
        per_file = pd.DataFrame(records, columns=cols)

        def row_stats(label, series, usl=0, lsl=0):
            mn = round(series.min(),2)
            mx = round(series.max(),2)
            mu = round(series.mean(),2)
            sigma = round(series.std(ddof=1),2)
            three = round(3 * sigma,2)  
            four5 = round(4.5 * sigma,2)

            if lsl != 0 and usl == 0:  # LSL only (like Return Loss)
                mean4p5Sigma = mu - four5  # Subtract for LSL
                mean3Sigma = mu - three    # Subtract for LSL
            elif lsl == 0 and usl != 0:
                mean4p5Sigma = mu + four5  # Add for USL
                mean3Sigma = mu + three    # Add for USL
            else:  # USL or both limits
                mean4p5Sigma = mu + four5  # Add for USL
                mean3Sigma = mu + three    # Add for USL

            usl_minus_mean = abs(usl - mu)   
            mean_minus_lsl = abs(mu - lsl)   
            c_hi = round(abs(usl_minus_mean / three),2) if three != 0 else 0
            c_lo = round(abs(mean_minus_lsl / three),2) if three != 0 else 0
            cpk = round(min(c_hi, c_lo),2) if three != 0 else 0

            return {
                "Parameter": label,
                "Min": mn,
                "Max": mx,
                "Mean": mu,
                "Sigma": sigma,
                "μ+/-4p5Sigma": mean4p5Sigma,
                "μ+/-3Sigma": mean3Sigma,
                "USL": usl,
                "LSL": lsl,
                "USL-Mean": usl_minus_mean,
                "USL-Mean/3sigma": c_hi,
                "Mean-LSL": mean_minus_lsl,
                "Mean-LSL/3sigma": c_lo,
                "CpK": cpk
            }

        summary_rows = []
        for kpi, bands in kpi_config["KPIs"].items():
            for b in bands:
                col = kpi + "_" + b["name"]
                summary_rows.append(row_stats(col, per_file[col], b.get("USL", 0), b.get("LSL", 0)))

        for sb in kpi_config["StopBands"]:
            col = sb["name"]
            summary_rows.append(row_stats(col, per_file[col], 0, sb["LSL"]))

        summary = pd.DataFrame(summary_rows)

        summary_excel_path = os.path.join(base_path, 'SParam_Summary.xlsx')
        with pd.ExcelWriter(summary_excel_path, engine='openpyxl') as xl:
            summary.to_excel(xl, sheet_name="Summary", index=False)
            per_file.to_excel(xl, sheet_name="Per_File", index=False)

        # Convert DataFrames to JSON-serializable dicts
        summary_json = summary.to_dict(orient='records')
        per_file_json = per_file.to_dict(orient='records')

        thelifi_logs.info(f"Summary Excel and JSON generated at {summary_excel_path}")
        return summary_excel_path, summary_json, per_file_json 

class SubmissionFilesListAPIView(APIView):
    """
    Get all files in the submission folder by submission_id.
    """
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Get all files in the submission folder by submission_id.",
        manual_parameters=[
            openapi.Parameter(
                'submission_id', openapi.IN_QUERY, description="ID of the uploaded submission record",
                type=openapi.TYPE_INTEGER, required=True
            )
        ],
        responses={200: "List of files", 400: "Bad Request", 404: "Submission not found"}
    )
    def get(self, request):
        """
        List all files in the submission folder.
        """
        try:
            submission_id = request.query_params.get('submission_id')

            if not submission_id:
                thelifi_logs.warning("File list fetch failed: submission_id missing.")
                return Response({'error': 'submission_id is required as a query parameter.'}, status=400)

            try:
                submission = FilterSubmission.objects.get(id=submission_id)
            except FilterSubmission.DoesNotExist:
                thelifi_logs.warning(f"File list fetch: Submission not found for id={submission_id}")
                return Response({'error': 'Submission not found.'}, status=404)

            folder_full_path = os.path.join(settings.MEDIA_ROOT, submission.folder_path)

            if not os.path.exists(folder_full_path):
                thelifi_logs.warning(f"File list fetch: Folder not found for submission_id={submission_id}")
                return Response({'error': 'Submission folder not found on server.'}, status=404)

            file_list = []
            for root, dirs, files in os.walk(folder_full_path):
                for file in files:
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, settings.MEDIA_ROOT)
                    file_list.append(f"/media/{relative_path.replace(os.path.sep, '/')}")

            thelifi_logs.info(f"File list fetched for submission_id={submission_id}")
            return Response({
                'message': 'File list fetched successfully.',
                'submission_id': submission_id,
                'submission_folder': submission.folder_path,
                'files': file_list
            }, status=200)

        except Exception as e:
            thelifi_logs.error(f"Error in SubmissionFilesListAPIView: {str(e)}")
            return Response({'error': str(e)}, status=500)

class DownloadPlotsZipAPIView(APIView):
    """
    API to download selected plots as a ZIP file.
    """
    permission_classes = [AllowAny]

    @swagger_auto_schema(
        operation_description="Download plots as a ZIP file by submission_id and plot type.",
        manual_parameters=[
            openapi.Parameter('submission_id', openapi.IN_QUERY, description="Submission ID", type=openapi.TYPE_INTEGER, required=True),
            openapi.Parameter('plots', openapi.IN_QUERY, description="Comma-separated plot types: sparmas, histogram, boxplot", type=openapi.TYPE_STRING, required=True),
        ],
        responses={200: "Zip File", 400: "Bad Request", 404: "Submission not found"}
    )
    def get(self, request):
        try:
            submission_id = request.query_params.get('submission_id')
            plots_param = request.query_params.get('plots')

            if not submission_id or not plots_param:
                thelifi_logs.warning("DownloadPlotsZipAPIView: Missing submission_id or plots parameter.")
                return Response({'error': 'submission_id and plots parameters are required.'}, status=400)

            plot_types = [plot.strip().lower() for plot in plots_param.split(',')]
            allowed_plot_types = ['sparmas', 'histogram', 'boxplot']

            for plot in plot_types:
                if plot not in allowed_plot_types:
                    thelifi_logs.warning(f"DownloadPlotsZipAPIView: Invalid plot type {plot}")
                    return Response({'error': f"Invalid plot type: {plot}. Allowed: {allowed_plot_types}"}, status=400)

            try:
                submission = FilterSubmission.objects.get(id=submission_id)
            except FilterSubmission.DoesNotExist:
                thelifi_logs.warning(f"DownloadPlotsZipAPIView: Submission not found for id={submission_id}")
                return Response({'error': 'Submission not found.'}, status=404)

            folder_full_path = os.path.join(settings.MEDIA_ROOT, submission.folder_path)

            if not os.path.exists(folder_full_path):
                thelifi_logs.warning(f"DownloadPlotsZipAPIView: Folder not found for submission_id={submission_id}")
                return Response({'error': 'Submission folder not found on server.'}, status=404)

            selected_files = []

            for root, dirs, files in os.walk(folder_full_path):
                # Focus only on 'Generated_Plots' folder
                if 'Generated_Plots' in root:
                    for file in files:
                        if (
                            ('boxplot' in plot_types and file.startswith('BoxPlot_')) or
                            ('histogram' in plot_types and file.startswith('Histogram_')) or
                            ('sparmas' in plot_types and not (file.startswith('BoxPlot_') or file.startswith('Histogram_')))
                        ):
                            selected_files.append(os.path.join(root, file))

            if not selected_files:
                thelifi_logs.warning(f"DownloadPlotsZipAPIView: No files found for selected plot types for submission_id={submission_id}")
                return Response({'error': 'No files found for the selected plot types.'}, status=404)

            # Create ZIP in memory
            import zipfile
            zip_buffer = io.BytesIO()
            zip_name = f"{submission.model_number}_{submission.edu_number}_{'_'.join(plot_types)}.zip"

            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for file_path in selected_files:
                    arcname = os.path.relpath(file_path, folder_full_path)
                    zip_file.write(file_path, arcname)

            zip_buffer.seek(0)
            thelifi_logs.info(f"DownloadPlotsZipAPIView: ZIP created for submission_id={submission_id}, plots={plot_types}")

            from django.http import FileResponse
            return FileResponse(zip_buffer, as_attachment=True, filename=zip_name)

        except Exception as e:
            thelifi_logs.error(f"Error in DownloadPlotsZipAPIView: {str(e)}")
            return Response({'error': str(e)}, status=500)

class SParameterPlotAPIView(APIView):
    @swagger_auto_schema(
        operation_description="Generate S-Parameter plots based on submission_id, plot configuration, and KPI configuration.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'submission_id': openapi.Schema(type=openapi.TYPE_INTEGER, description="ID of the submission"),
                'plot_config': openapi.Schema(type=openapi.TYPE_OBJECT, description="Plot configuration JSON"),
                'kpi_config_data': openapi.Schema(type=openapi.TYPE_OBJECT, description="KPI configuration JSON (must include 'KPIs' key)")
            },
            required=['submission_id', 'plot_config', 'kpi_config_data']
        ),
        responses={200: "Plots generated successfully", 400: "Bad Request", 404: "Submission not found"}
    )
    def post(self, request):
        try:
            submission_id = request.data.get('submission_id')
            plot_config = request.data.get('plot_config')
            kpi_config_data = request.data.get('kpi_config_data')

            if not submission_id:
                thelifi_logs.warning("SParameterPlotAPIView: submission_id is required.")
                return Response({'error': 'submission_id is required.'}, status=400)
            if not plot_config:
                thelifi_logs.warning("SParameterPlotAPIView: plot_config is required.")
                return Response({'error': 'plot_config is required.'}, status=400)
            if not kpi_config_data or 'KPIs' not in kpi_config_data:
                thelifi_logs.warning("SParameterPlotAPIView: kpi_config_data missing or invalid.")
                return Response({'error': "kpi_config_data is required and must contain a 'KPIs' key."}, status=400)

            try:
                submission = FilterSubmission.objects.get(id=submission_id)
            except FilterSubmission.DoesNotExist:
                thelifi_logs.warning(f"SParameterPlotAPIView: Submission not found for id={submission_id}")
                return Response({'error': 'Submission not found.'}, status=404)

            folder_full_path = os.path.join(settings.MEDIA_ROOT, submission.folder_path)
            excel_files = glob.glob(os.path.join(folder_full_path, 'SParam_Summary.xlsx'))
            s2p_files = glob.glob(os.path.join(folder_full_path, 's2pFiles', '*.s2p'))
            sim_s2p_files = glob.glob(os.path.join(folder_full_path, 'Simulation_Files', '*_Simulated_*.s2p'))
            s11_sigma_files = glob.glob(os.path.join(folder_full_path, 'Simulation_Files', 'Simulated S11_Sigma_Data.csv'))
            s21_sigma_files = glob.glob(os.path.join(folder_full_path, 'Simulation_Files', 'Simulated S21_Sigma_Data.csv'))

            generated_plots_folder = os.path.join(folder_full_path, 'Generated_Plots')
            os.makedirs(generated_plots_folder, exist_ok=True)

            generated_plots = generate_s_parameter_plots_only(
                plot_config_data=plot_config,
                excel_files=excel_files,
                s2p_files=s2p_files,
                sim_s2p_files=sim_s2p_files,
                s11_sigma_files=s11_sigma_files,
                s21_sigma_files=s21_sigma_files,
                kpi_config_data=kpi_config_data,
                save_folder=generated_plots_folder
            )

            plot_urls = []
            for plot in generated_plots:
                plot_path = os.path.join(generated_plots_folder, plot)
                if os.path.exists(plot_path):
                    relative_plot_path = os.path.relpath(plot_path, settings.MEDIA_ROOT)
                    plot_urls.append(f"/media/{relative_plot_path.replace(os.path.sep, '/')}")

            thelifi_logs.info(f"S-Parameter plots generated for submission_id={submission_id}")
            return Response({'message': 'S-Parameter Plots Generated Successfully', 'generated_plots': plot_urls}, status=200)

        except FilterSubmission.DoesNotExist:
            thelifi_logs.warning("SParameterPlotAPIView: Submission not found in exception.")
            return Response({'error': 'Submission not found.'}, status=404)
        except Exception as e:
            thelifi_logs.error(f"Error in SParameterPlotAPIView: {str(e)}")
            return Response({'error': str(e)}, status=500)

class StatisticalAndHistogramPlotAPIView(APIView):
    @swagger_auto_schema(
        operation_description="Generate Statistical and Histogram plots based on submission_id and plot configuration.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'submission_id': openapi.Schema(type=openapi.TYPE_INTEGER, description="ID of the submission"),
                'plot_config': openapi.Schema(type=openapi.TYPE_OBJECT, description="Plot configuration JSON")
            },
            required=['submission_id', 'plot_config']
        ),
        responses={200: "Plots generated successfully", 400: "Bad Request", 404: "Submission not found"}
    )
    def post(self, request):
        try:
            submission_id = request.data.get('submission_id')
            plot_config = request.data.get('plot_config')

            if not submission_id:
                thelifi_logs.warning("StatisticalAndHistogramPlotAPIView: submission_id is required.")
                return Response({'error': 'submission_id is required.'}, status=400)

            if not plot_config:
                thelifi_logs.warning("StatisticalAndHistogramPlotAPIView: plot_config is required.")
                return Response({'error': 'plot_config is required.'}, status=400)

            submission = FilterSubmission.objects.get(id=submission_id)
            folder_full_path = os.path.join(settings.MEDIA_ROOT, submission.folder_path)
            excel_files = glob.glob(os.path.join(folder_full_path, 'SParam_Summary.xlsx'))

            generated_plots_folder = os.path.join(folder_full_path, 'Generated_Plots')
            os.makedirs(generated_plots_folder, exist_ok=True)

            generated_plots = generate_statistical_and_histogram_plots_only(
                plot_config_data=plot_config,
                excel_files=excel_files,
                save_folder=generated_plots_folder
            )

            plot_urls = []
            for plot in generated_plots:
                plot_path = os.path.join(generated_plots_folder, plot)
                if os.path.exists(plot_path):
                    relative_plot_path = os.path.relpath(plot_path, settings.MEDIA_ROOT)
                    plot_urls.append(f"/media/{relative_plot_path.replace(os.path.sep, '/')}")

            thelifi_logs.info(f"Statistical and Histogram plots generated for submission_id={submission_id}")
            return Response({'message': 'Statistical and Histogram Plots Generated Successfully', 'generated_plots': plot_urls}, status=200)

        except FilterSubmission.DoesNotExist:
            thelifi_logs.warning("StatisticalAndHistogramPlotAPIView: Submission not found.")
            return Response({'error': 'Submission not found.'}, status=404)
        except Exception as e:
            thelifi_logs.error(f"Error in StatisticalAndHistogramPlotAPIView: {str(e)}")
            return Response({'error': str(e)}, status=500)

class SubmissionRecordListAPIView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        try:
            records = FilterSubmission.objects.all().order_by('-created_at')
            serializer = FilterSubmissionSerializer(records, many=True)
            thelifi_logs.info("Fetched all submission records.")
            return Response(serializer.data, status=200)
        except Exception as e:
            thelifi_logs.error(f"Error in SubmissionRecordListAPIView: {str(e)}")
            return Response({'error': str(e)}, status=500)

class SubmissionRecordDetailAPIView(APIView):
    permission_classes = [AllowAny]

    def get(self, request, pk):
        try:
            record = FilterSubmission.objects.get(pk=pk)
            serializer = FilterSubmissionSerializer(record)
            thelifi_logs.info(f"Fetched submission record id={pk}")
            return Response(serializer.data, status=200)
        except FilterSubmission.DoesNotExist:
            thelifi_logs.warning(f"SubmissionRecordDetailAPIView: Record not found for id={pk}")
            return Response({'error': 'Record not found.'}, status=404)
        except Exception as e:
            thelifi_logs.error(f"Error in SubmissionRecordDetailAPIView (GET): {str(e)}")
            return Response({'error': str(e)}, status=500)

    def delete(self, request, pk):
        try:
            record = FilterSubmission.objects.get(pk=pk)

            # Get the full folder path
            folder_full_path = os.path.join(settings.MEDIA_ROOT, record.folder_path)

            # Delete the folder if it exists
            if os.path.exists(folder_full_path):
                shutil.rmtree(folder_full_path)

            # Delete the record from the database
            record.delete()

            thelifi_logs.info(f"Deleted submission record and folder for id={pk}")
            return Response({'message': 'Record and folder deleted successfully.'}, status=204)

        except FilterSubmission.DoesNotExist:
            thelifi_logs.warning(f"SubmissionRecordDetailAPIView: Record not found for id={pk}")
            return Response({'error': 'Record not found.'}, status=404)
        except Exception as e:
            thelifi_logs.error(f"Error in SubmissionRecordDetailAPIView (DELETE): {str(e)}")
            return Response({'error': str(e)}, status=500)

