from django.contrib import admin
from .models import Asset, Notification, Violation

admin.site.register(Asset)
admin.site.register(Notification)
admin.site.register(Violation)
