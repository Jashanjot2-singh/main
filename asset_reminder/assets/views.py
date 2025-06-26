from rest_framework import viewsets
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import Asset, Notification, Violation
from .serializers import AssetSerializer, NotificationSerializer, ViolationSerializer
from .tasks import run_checks

class AssetViewSet(viewsets.ModelViewSet):
    queryset = Asset.objects.all()
    serializer_class = AssetSerializer

class NotificationViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Notification.objects.all()
    serializer_class = NotificationSerializer

class ViolationViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Violation.objects.all()
    serializer_class = ViolationSerializer

class RunChecksView(APIView):
    def post(self, request):
        run_checks()
        return Response({"detail": "Checks completed."})

