from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    """Extended user profile for PulmoScan"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    title = models.CharField(max_length=100, blank=True, null=True)
    institution = models.CharField(max_length=200, blank=True, null=True)
    specialty = models.CharField(max_length=100, blank=True, null=True)
    profile_image = models.ImageField(upload_to='profile_images/', blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username}'s Profile"
