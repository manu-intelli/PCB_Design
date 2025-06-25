import os
import uuid
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from .models import FilterSubmission

class FilterUploadView(APIView):
    parser_classes = [MultiPartParser, FormParser]  # 🔥 Mandatory for file upload

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
        model_number = request.data.get('model_number')
        edu_number = request.data.get('edu_number')
        filter_type = request.data.get('filter_type')
        s2p_files = request.FILES.getlist('s2p_files')
        simulation_files = request.FILES.getlist('simulation_files')

        if not (model_number and edu_number and filter_type and s2p_files):
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

        FilterSubmission.objects.create(
            folder_name=folder_name,
            model_number=model_number,
            edu_number=edu_number,
            filter_type=filter_type
        )

        return Response({"message": "Files uploaded successfully.", "folder_name": folder_name}, status=status.HTTP_201_CREATED)


import os
import glob
import warnings
import numpy as np
import pandas as pd
import skrf as rf
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from .models import FilterSubmission


class KPI_CalculationAPIView(APIView):
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
        submission_id = request.data.get('submission_id')
        kpi_data = request.data.get('kpi_data')

        if not submission_id or not kpi_data:
            return Response({"error": "submission_id and kpi_data are required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            submission = FilterSubmission.objects.get(id=submission_id)
        except FilterSubmission.DoesNotExist:
            return Response({"error": "Submission not found."}, status=status.HTTP_404_NOT_FOUND)

        # File paths
        base_path = os.path.join(settings.MEDIA_ROOT, 'uploads', submission.folder_name)
        s2p_path = os.path.join(base_path, 's2pFiles')
        plot_path = os.path.join(base_path, 'Generated_Plots')

        try:
            excel_path = self.generate_summary(s2p_path, plot_path, kpi_data)

            return Response({
                "message": "Summary generated successfully.",
                "excel_file_path": excel_path.replace(settings.MEDIA_ROOT, '/media')
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def generate_summary(self, s2p_folder, plot_folder, kpi_config):
        records = []
        s2p_files = glob.glob(os.path.join(s2p_folder, 'U*.s2p'))

        if not s2p_files:
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
            gd_ns = -np.gradient(phase, f) / (2 * np.pi) * 1e9

            out = []
            for kpi, bands in CFG["KPIs"].items():
                for b in bands:
                    lo, hi = b["range"]
                    mask = (f >= lo) & (f <= hi)

                    if kpi == "IL":
                        val = -mag21[mask].min()
                    elif kpi == "RL":
                        val = min(-mag11[mask].max(), -mag22[mask].max())
                    elif kpi == "Flat":
                        val = mag21[mask].max() - mag21[mask].min()
                    elif kpi == "GD":
                        val = gd_ns[mask].max()
                    elif kpi == "GDV":
                        val = np.ptp(gd_ns[mask])
                    elif kpi == "LPD":
                        A = np.vstack([f[mask], np.ones_like(f[mask])]).T
                        slope, intercept = np.linalg.lstsq(A, phase[mask], rcond=None)[0]
                        resid = phase - (slope * f + intercept)
                        lpd_deg = np.degrees(resid)
                        val = max(abs(lpd_deg[mask].max()), abs(lpd_deg[mask].min()))
                    else:
                        val = np.nan
                    out.append(val)

            for sb in CFG["StopBands"]:
                lo, hi = sb["range"]
                rej = -mag21[(f >= lo) & (f <= hi)].max()
                out.append(rej)
            return out

        # Start processing
        for p in s2p_files:
            cp = convert_type2(p)
            try:
                nt = rf.Network(cp)
            except Exception as e:
                warnings.warn("Skipping " + p + ": " + str(e))
                continue
            records.append([os.path.basename(p)] + kpis_for_network(nt, kpi_config))

        cols = ["File"]
        for kpi, bands in kpi_config["KPIs"].items():
            cols.extend([kpi + "_" + b["name"] for b in bands])
        cols.extend([sb["name"] for sb in kpi_config["StopBands"]])
        per_file = pd.DataFrame(records, columns=cols)

        def row_stats(label, series, usl=np.nan, lsl=np.nan):
            mn = series.min()
            mx = series.max()
            mu = series.mean()
            sigma = series.std(ddof=0)
            three = 3 * sigma
            four5 = 4.5 * sigma

            usl_minus_mean = usl - mu if not np.isnan(usl) else np.nan
            mean_minus_lsl = mu - lsl if not np.isnan(lsl) else np.nan
            c_hi = usl_minus_mean / three if not np.isnan(usl) else np.nan
            c_lo = mean_minus_lsl / three if not np.isnan(lsl) else np.nan
            cpk = min(c_hi, c_lo) if not (np.isnan(c_hi) or np.isnan(c_lo)) else np.nan

            return dict(
                Parameter=label, Min=mn, Max=mx, Mean=mu, Sigma=sigma,
                _4p5Sigma=four5, _3Sigma=three,
                USL=usl, LSL=lsl,
                USL_Mean=usl_minus_mean, _USL_Mean_div_3sigma=c_hi,
                Mean_LSL=mean_minus_lsl, _Mean_LSL_div_3sigma=c_lo,
                CpK=cpk
            )

        summary_rows = []
        for kpi, bands in kpi_config["KPIs"].items():
            for b in bands:
                col = kpi + "_" + b["name"]
                summary_rows.append(row_stats(col, per_file[col], b.get("USL", np.nan), b.get("LSL", np.nan)))

        for sb in kpi_config["StopBands"]:
            col = sb["name"]
            summary_rows.append(row_stats(col, per_file[col], np.nan, sb["LSL"]))

        summary = pd.DataFrame(summary_rows)

        summary_excel_path = os.path.join(s2p_folder, 'SParam_Summary.xlsx')
        with pd.ExcelWriter(summary_excel_path, engine='openpyxl') as xl:
            summary.to_excel(xl, sheet_name="Summary", index=False)
            per_file.to_excel(xl, sheet_name="Per_File", index=False)

        return summary_excel_path
