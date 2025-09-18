from django.contrib import admin
from .models import Profile

# Register your models here.
@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'dark_mode', 'email_notifications')
    search_fields = ('user_username','user_email')