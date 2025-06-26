from django.db import models

class Asset(models.Model):
    name = models.CharField(max_length=100)
    service_time = models.DateTimeField()
    expiration_time = models.DateTimeField()
    is_serviced = models.BooleanField(default=False)

class Notification(models.Model):
    NOTIF_TYPES = [('service', 'Service'), ('expiration', 'Expiration')]
    asset = models.ForeignKey(Asset, on_delete=models.CASCADE)
    type = models.CharField(max_length=10, choices=NOTIF_TYPES)
    timestamp = models.DateTimeField(auto_now_add=True)

class Violation(models.Model):
    VIOLATION_TYPES = [('service', 'Service'), ('expiration', 'Expiration')]
    asset = models.ForeignKey(Asset, on_delete=models.CASCADE)
    type = models.CharField(max_length=10, choices=VIOLATION_TYPES)
    timestamp = models.DateTimeField(auto_now_add=True)