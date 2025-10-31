from django.apps import AppConfig
from django.conf import settings
from django.contrib.auth import get_user_model
from django.db.utils import OperationalError, ProgrammingError
from django.db.models.signals import post_migrate


class HospitalWebappConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'hospital_webapp'
    label = 'webapp'  # present as 'webapp' while module is hospital_webapp

    def ready(self):
        # Ensure default admin is created after migrations
        def ensure_admin(sender, **kwargs):
            try:
                User = get_user_model()
                if not User.objects.filter(username='admin').exists():
                    User.objects.create_superuser('admin', 'admin@example.com', 'admin123')
            except Exception:
                pass

        post_migrate.connect(ensure_admin, sender=self)
