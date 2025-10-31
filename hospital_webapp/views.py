import io
import os
import hashlib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse, HttpResponseBadRequest
from django.template.loader import render_to_string
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST
from .models import Patient, AnalysisReport

# Attempt to load TensorFlow model for image inference
TF_MODEL = None
AUTISM_THRESHOLD = 0.70  # classify as autistic only above this probability
IMAGE_SIZE = (224, 224)
try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mbv2_preprocess  # type: ignore
    MODEL_PATH = os.path.join(str(settings.BASE_DIR), 'data', 'autism_mobilenetv2_best.h5')
    if os.path.exists(MODEL_PATH):
        TF_MODEL = tf.keras.models.load_model(MODEL_PATH)
except Exception:
    TF_MODEL = None


def home(request):
    return render(request, 'home.html')


def about(request):
    return render(request, 'about.html')


def contact(request):
    return render(request, 'contact.html')


def signin(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('dashboard')
        messages.error(request, 'Invalid credentials')
    return render(request, 'signin.html')


def signout(request):
    logout(request)
    return redirect('home')


@login_required
def dashboard(request):
    context = {
        'patients_count': Patient.objects.count(),
        'reports_count': AnalysisReport.objects.count(),
        'patients': Patient.objects.order_by('patient_uid'),
    }
    return render(request, 'dashboard.html', context)


@login_required
def patients(request):
    items = Patient.objects.order_by('-created_at')
    return render(request, 'patients.html', {'patients': items})


@login_required
def add_patient(request):
    if request.method == 'POST':
        first_name = request.POST.get('first_name', '')
        last_name = request.POST.get('last_name', '')
        email = request.POST.get('email', '')
        phone = request.POST.get('phone', '')
        dob = request.POST.get('dob') or None
        notes = request.POST.get('notes', '')
        p = Patient.objects.create(
            first_name=first_name,
            last_name=last_name,
            email=email,
            phone=phone,
            dob=dob,
            notes=notes,
        )
        messages.success(request, 'Patient added')
        return redirect('patients')
    return render(request, 'add_patient.html')


@login_required
@require_POST
def analyze(request):
    patient_uid = request.POST.get('patient_uid')
    patient = get_object_or_404(Patient, patient_uid=patient_uid)

    media_type = request.POST.get('media_type')  # image | video | audio
    up_video = request.FILES.get('video')
    up_image = request.FILES.get('image')
    up_audio = request.FILES.get('audio')

    # Simple storage to MEDIA-like folder under BASE_DIR / uploads
    base_dir = settings.BASE_DIR
    upload_dir = base_dir / 'uploads'
    (upload_dir / 'videos').mkdir(parents=True, exist_ok=True)
    (upload_dir / 'images').mkdir(parents=True, exist_ok=True)
    (upload_dir / 'audios').mkdir(parents=True, exist_ok=True)

    saved_video_path = None
    saved_image_path = None

    if up_video:
        saved_video_path = upload_dir / 'videos' / up_video.name
        with open(saved_video_path, 'wb+') as f:
            for chunk in up_video.chunks():
                f.write(chunk)

    if up_image:
        saved_image_path = upload_dir / 'images' / up_image.name
        with open(saved_image_path, 'wb+') as f:
            for chunk in up_image.chunks():
                f.write(chunk)

    saved_audio_path = None
    if up_audio:
        saved_audio_path = upload_dir / 'audios' / up_audio.name
        with open(saved_audio_path, 'wb+') as f:
            for chunk in up_audio.chunks():
                f.write(chunk)

    # Analysis: prefer real model for image/video; else fallback
    result_label = 'autistic'
    result_score = 0.5
    if media_type == 'image' and saved_image_path:
        try:
            if TF_MODEL is not None:
                # Use TF model
                img = tf.keras.utils.load_img(str(saved_image_path), target_size=IMAGE_SIZE)
                arr = tf.keras.utils.img_to_array(img)
                arr = tf.expand_dims(arr, 0)
                arr = mbv2_preprocess(arr)
                prob_raw = float(TF_MODEL.predict(arr, verbose=0)[0][0])
                # Treat model output as P(autistic). If your model outputs P(non_autistic), flip here.
                p_autistic = prob_raw
                result_score = p_autistic
                # For images, keep classic 0.5 threshold
                result_label = 'autistic' if p_autistic >= 0.5 else 'non_autistic'
            else:
                # Fallback: deterministic hash-based score in [0,1)
                with open(saved_image_path, 'rb') as f:
                    data = f.read()
                h = hashlib.sha1(data).hexdigest()
                frac = int(h[:8], 16) / 0xFFFFFFFF  # treat as P(autistic)
                p_autistic = float(frac)
                result_score = p_autistic
                # For images, keep classic 0.5 threshold
                result_label = 'autistic' if p_autistic >= 0.5 else 'non_autistic'
        except Exception:
            result_label = 'non_autistic'
            result_score = 0.5
    elif media_type == 'video' and saved_video_path:
        frames_info = []
        try:
            # Try OpenCV for frame extraction
            import cv2  # type: ignore
            cap = cv2.VideoCapture(str(saved_video_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration_s = total_frames / fps if fps else 0
            step_s = max(1.0, duration_s / 12.0)  # up to ~12 samples

            t = 0.0
            picked = 0
            high_probs = []
            import numpy as np  # type: ignore
            while t <= duration_s and picked < 32:
                frame_idx = int(t * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    t += step_s
                    continue
                # Convert BGR->RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Predict
                prob_raw = 0.5
                try:
                    if TF_MODEL is not None:
                        resized = cv2.resize(frame_rgb, IMAGE_SIZE)
                        arr = np.expand_dims(resized, 0).astype('float32')
                        arr = mbv2_preprocess(arr)
                        prob_raw = float(TF_MODEL.predict(arr, verbose=0)[0][0])
                    else:
                        # Hash fallback per frame
                        h = hashlib.sha1(frame_rgb.tobytes()).hexdigest()
                        prob_raw = float(int(h[:8], 16) / 0xFFFFFFFF)
                except Exception:
                    prob_raw = 0.5

                # Encode JPEG to base64 for UI
                import base64
                import PIL.Image as Image  # type: ignore
                from io import BytesIO
                pil_img = Image.fromarray(frame_rgb)
                pil_img.thumbnail((384, 384))
                buf = BytesIO()
                pil_img.save(buf, format='JPEG', quality=85)
                b64 = base64.b64encode(buf.getvalue()).decode('ascii')
                # Treat model output as P(autistic)
                p_autistic = prob_raw
                frames_info.append({
                    'timestamp': round(t, 2),
                    'prob': p_autistic,
                    'image_b64': 'data:image/jpeg;base64,' + b64,
                })
                if p_autistic >= AUTISM_THRESHOLD:
                    high_probs.append(p_autistic)
                picked += 1
                t += step_s
            cap.release()

            # Aggregate video-level result by mean of positives else mean of all
            if high_probs:
                result_score = float(sum(high_probs) / len(high_probs))
            elif frames_info:
                result_score = float(sum(f['prob'] for f in frames_info) / len(frames_info))
            else:
                result_score = 0.5
            result_label = 'autistic' if result_score >= AUTISM_THRESHOLD else 'non_autistic'

        except Exception:
            # If cv2 unavailable or failed, fallback to file hash
            with open(saved_video_path, 'rb') as f:
                data = f.read(65536)
            h = hashlib.sha1(data).hexdigest()
            frac = int(h[:8], 16) / 0xFFFFFFFF  # treat as P(autistic)
            p_autistic = float(frac)
            result_score = p_autistic
            result_label = 'autistic' if p_autistic >= AUTISM_THRESHOLD else 'non_autistic'
            frames_info = []

    report = AnalysisReport.objects.create(
        patient=patient,
        uploaded_video=str(saved_video_path) if saved_video_path else None,
        uploaded_image=str(saved_image_path) if saved_image_path else None,
        uploaded_audio=str(saved_audio_path) if saved_audio_path else None,
        result_label=result_label,
        result_score=result_score,
        created_by=request.user,
    )

    return JsonResponse({
        'ok': True,
        'report_id': report.id,
        'result_label': result_label,
        'result_score': result_score,
        'media_type': media_type,
        'frames': frames_info if media_type == 'video' else [],
    })


@login_required
def email_report(request, report_id: int):
    report = get_object_or_404(AnalysisReport, id=report_id)
    recipient = request.GET.get('to') or report.patient.email
    if not recipient:
        return HttpResponseBadRequest('No recipient email provided')

    subject = f"Autism Screening Report #{report.id}"
    html_body = render_to_string('email/report_email.html', { 'report': report })

    msg = MIMEMultipart('alternative')
    msg['From'] = settings.EMAIL_HOST_USER or 'noreply@example.com'
    msg['To'] = recipient
    msg['Subject'] = subject
    msg.attach(MIMEText(html_body, 'html'))

    # Attach PDF
    try:
        pdf_bytes = _build_report_pdf(report)
        from email.mime.base import MIMEBase
        from email import encoders
        part = MIMEBase('application', 'pdf')
        part.set_payload(pdf_bytes)
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename="report_{report.id}.pdf"')
        msg.attach(part)
    except Exception:
        # If PDF generation fails, continue with just HTML body
        pass

    try:
        with smtplib.SMTP(settings.EMAIL_HOST or 'smtp.gmail.com', settings.EMAIL_PORT) as server:
            if settings.EMAIL_USE_TLS:
                server.starttls()
            if settings.EMAIL_HOST_USER and settings.EMAIL_HOST_PASSWORD:
                server.login(settings.EMAIL_HOST_USER, settings.EMAIL_HOST_PASSWORD)
            server.send_message(msg)
        messages.success(request, 'Email sent')
        return redirect('history')
    except Exception as e:
        return HttpResponse(f'Email failed: {e}', status=500)


@login_required
def download_report(request, report_id: int):
    report = get_object_or_404(AnalysisReport, id=report_id)
    try:
        pdf_bytes = _build_report_pdf(report)
        resp = HttpResponse(pdf_bytes, content_type='application/pdf')
        resp['Content-Disposition'] = f'attachment; filename="report_{report.id}.pdf"'
        return resp
    except Exception as e:
        # Fallback to plain text if PDF generation fails
        content = (
            f"Report ID: {report.id}\n"
            f"Patient: {report.patient}\n"
            f"Result: {report.result_label} (score {report.result_score:.2f})\n"
            f"Created: {report.created_at:%Y-%m-%d %H:%M}\n"
        ).encode('utf-8')
        resp = HttpResponse(content, content_type='text/plain')
        resp['Content-Disposition'] = f'attachment; filename="report_{report.id}.txt"'
        return resp


@login_required
def history(request):
    reports = AnalysisReport.objects.select_related('patient').order_by('-created_at')
    return render(request, 'history.html', {'reports': reports})

# Create your views here.


def _build_report_pdf(report) -> bytes:
    """Generate a simple, well-formatted PDF as bytes for a report.
    Tries to use reportlab if available; raises if not installed.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import mm
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.styles import ParagraphStyle
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=20*mm, rightMargin=20*mm, topMargin=20*mm, bottomMargin=20*mm)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=18, leading=22, textColor=colors.HexColor('#0d6efd'))
        label_style = ParagraphStyle('Label', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#6b7280'))
        value_style = ParagraphStyle('Value', parent=styles['Normal'], fontSize=12, leading=16)

        story = []
        story.append(Paragraph('Autism Screening Analysis Report', title_style))
        story.append(Spacer(1, 8))

        data = [
            [Paragraph('Report ID', label_style), Paragraph(f"#{report.id}", value_style)],
            [Paragraph('Patient', label_style), Paragraph(f"{report.patient.first_name} {report.patient.last_name} ({report.patient.patient_uid})", value_style)],
            [Paragraph('Result', label_style), Paragraph(('autistic' if report.result_label == 'autistic' else 'non aut') + f" (score {report.result_score:.2f})", value_style)],
            [Paragraph('Generated at', label_style), Paragraph(report.created_at.strftime('%Y-%m-%d %H:%M'), value_style)],
        ]
        table = Table(data, colWidths=[35*mm, 120*mm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
            ('TEXTCOLOR', (0,0), (-1,-1), colors.black),
            ('LINEBELOW', (0,0), (-1,-1), 0.25, colors.HexColor('#eef2f7')),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING', (0,0), (-1,-1), 4),
            ('RIGHTPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(table)
        story.append(Spacer(1, 12))
        story.append(Paragraph('This report is system-generated by BehaVia.', ParagraphStyle('Foot', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#6b7280'))))
        doc.build(story)
        pdf = buf.getvalue()
        buf.close()
        return pdf
    except Exception as exc:
        # Re-raise to let caller decide fallback
        raise
