from django.contrib import admin
from .models import Patient, AnalysisReport


@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ("first_name", "last_name", "email", "phone", "created_at")
    search_fields = ("first_name", "last_name", "email", "phone")


@admin.register(AnalysisReport)
class AnalysisReportAdmin(admin.ModelAdmin):
    list_display = ("id", "patient", "result_label", "result_score", "created_at")
    list_filter = ("result_label",)
    search_fields = ("patient__first_name", "patient__last_name")

# Register your models here.
