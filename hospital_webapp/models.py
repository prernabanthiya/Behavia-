from django.db import models
from django.contrib.auth import get_user_model
from django.db.models import Max


class Patient(models.Model):
    patient_uid = models.CharField(max_length=16, unique=True, blank=True)
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    email = models.EmailField(blank=True)
    phone = models.CharField(max_length=20, blank=True)
    dob = models.DateField(null=True, blank=True)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.patient_uid} - {self.first_name} {self.last_name}".strip()

    def save(self, *args, **kwargs):
        if not self.patient_uid:
            # Generate next UID like P0001
            last = Patient.objects.aggregate(max_uid=Max('patient_uid'))['max_uid']
            next_num = 1
            if last and last.startswith('P') and last[1:].isdigit():
                next_num = int(last[1:]) + 1
            self.patient_uid = f"P{next_num:04d}"
        super().save(*args, **kwargs)


class AnalysisReport(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='reports')
    uploaded_video = models.FileField(upload_to='uploads/videos/', blank=True, null=True)
    uploaded_image = models.ImageField(upload_to='uploads/images/', blank=True, null=True)
    uploaded_audio = models.FileField(upload_to='uploads/audios/', blank=True, null=True)
    result_label = models.CharField(max_length=64)
    result_score = models.FloatField(default=0.0)
    created_by = models.ForeignKey(get_user_model(), on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    notes = models.TextField(blank=True)

    def __str__(self):
        return f"Report #{self.id} - {self.patient} - {self.result_label}"

# Create your models here.
