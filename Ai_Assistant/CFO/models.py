from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import os
from django.dispatch import receiver
from django.db.models.signals import post_save


class FinancialReport(models.Model):
    RISK_CHOICES = [
        ("High Risk", "High Risk"),
        ("Moderate/OK", "Moderate/OK"),
        ("Low/Healthy", "Low/Healthy"),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, default=1)  # default user
    title = models.CharField(max_length=255, blank=True, null=True, default="Untitled Report")
    file = models.FileField(upload_to='reports/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    size = models.BigIntegerField(null=True, blank=True, help_text="File size in bytes")

    # KPI Metrics
    total_revenue = models.FloatField(null=True, blank=True)
    total_expense = models.FloatField(null=True, blank=True)
    total_profit = models.FloatField(null=True, blank=True)
    cash_balance = models.FloatField(null=True, blank=True)
    revenue_growth = models.FloatField(null=True, blank=True, help_text="Percentage value (e.g., 20 for 20%)")
    profit_margin = models.FloatField(null=True, blank=True, help_text="Percentage value (e.g., 15 for 15%)")
    burn_rate = models.FloatField(null=True, blank=True, help_text="Monthly outflow in $")
    runway_months = models.FloatField(null=True, blank=True, help_text="Months of runway")

    # Risk
    risk_flag = models.CharField(max_length=50, choices=RISK_CHOICES, null=True, blank=True)

    def __str__(self):
        return self.title or os.path.basename(self.file.name)


class FinancialGoal(models.Model):
    GOAL_TYPES = [
        ("revenue", "Revenue"),
        ("profit", "Profit"),
        ("expense", "Expense"),
        ("cash", "Cash Balance"),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    goal_type = models.CharField(max_length=20, choices=GOAL_TYPES)
    target_value = models.FloatField()
    deadline = models.DateField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.get_goal_type_display()} ({self.target_value})"



class Contact(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def _str_(self):
        return f"{self.name} ({self.email})"
    

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    dark_mode = models.BooleanField(default=False)
    email_notifications = models.BooleanField(default=True)

    def _str_(self):
        return f"Profile: {self.user.username}"

@receiver(post_save, sender=User)
def ensure_profile_exists(sender, instance, created, **kwargs):
    # create profile automatically for new users
    if created:
        Profile.objects.create(user=instance)


class UserPreferences(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    email_notifications = models.BooleanField(default=True)
    
    def _str_(self):
        return f"{self.user.username}'s Preferences"
    

