from rest_framework import serializers
from .models import Asset, Notification, Violation

class AssetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Asset
        fields = '__all__'

    def validate(self, data):
        if data['service_time'] > data['expiration_time']:
            raise serializers.ValidationError("Service time must be before expiration time")
        return data

class NotificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Notification
        fields = '__all__'

class ViolationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Violation
        fields = '__all__'