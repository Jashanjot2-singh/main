from datetime import timedelta
from plyer import notification
import time
def run_checks():
    from django.utils import timezone
    from .models import Asset, Notification, Violation

    now = timezone.now()
    delta = timedelta(minutes=15)

    for asset in Asset.objects.all():
        if not asset.is_serviced:
            # Service time reminder
            time_diff = asset.service_time - now
            if timedelta(0) < time_diff <= delta:
                notification.notify(title=f"Service: {asset.name}", message="Service due soon.", timeout=5)
                Notification.objects.get_or_create(asset=asset, type='service')

            # Missed service
            if now >= asset.service_time:
                Violation.objects.get_or_create(asset=asset, type='service')

            # Expiration reminder
            exp_diff = asset.expiration_time - now
            if timedelta(0) < exp_diff <= delta:
                notification.notify(title=f"Expiration: {asset.name}", message="Expires soon.", timeout=5)
                Notification.objects.get_or_create(asset=asset, type='expiration')

            # Missed expiration
            if now >= asset.expiration_time:
                Violation.objects.get_or_create(asset=asset, type='expiration')

def run_checks_loop():
    while True:
        run_checks()
        time.sleep(60)