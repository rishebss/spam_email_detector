from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views

app_name = "detectorapp"

urlpatterns = [
    path('', views.home, name="home"),

]