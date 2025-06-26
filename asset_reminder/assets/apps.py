from django.apps import AppConfig
import threading
from .tasks import run_checks_loop

class AssetsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'assets'

    def ready(self):
        threading.Thread(target=run_checks_loop, daemon=True).start()
