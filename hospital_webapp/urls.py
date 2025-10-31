from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path('signin/', views.signin, name='signin'),
    path('signout/', views.signout, name='signout'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('patients/', views.patients, name='patients'),
    path('patients/add/', views.add_patient, name='add_patient'),
    path('analyze/', views.analyze, name='analyze'),  # AJAX endpoint
    path('reports/<int:report_id>/email/', views.email_report, name='email_report'),
    path('reports/<int:report_id>/download/', views.download_report, name='download_report'),
    path('history/', views.history, name='history'),
]


