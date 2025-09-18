from django import forms
from .models import FinancialReport, FinancialGoal, Profile
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm, PasswordChangeForm
from django import forms


class FinancialReportForm(forms.ModelForm):
    class Meta:
        model = FinancialReport
        fields = ['title', 'file']
        widgets = {
            'title': forms.TextInput(attrs={'placeholder': 'Enter report title (optional)'}),
        }
class SignUpForm(UserCreationForm):
    class Meta:
        model = User
        fields = ["username", "email", "password1", "password2"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs.update({"class": "form-control rounded-3 py-2"})

            
class LoginForm(AuthenticationForm):
    username = forms.CharField(
        widget=forms.TextInput(attrs={
            "class": "form-control",
            "placeholder": "Enter your username"
        })
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            "class": "form-control",
            "placeholder": "Enter your password"
        })
    )

class ScenarioForm(forms.Form):
    revenue_growth = forms.FloatField(
        required=False, label="Revenue Growth (%)",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g. 10'})
    )
    expense_growth = forms.FloatField(
        required=False, label="Expense Growth (%)",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g. 5'})
    )
    cash_infusion = forms.FloatField(
        required=False, label="Cash Infusion (₹)",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g. 50000'})
    )
    months = forms.IntegerField(
        required=False, label="Projection Months",
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g. 6'})
    )


class ReportFilterForm(forms.Form):
    start_date = forms.DateField(
        required=False, widget=forms.DateInput(attrs={'type':'date', 'class':'form-control'})
    )
    end_date = forms.DateField(
        required=False, widget=forms.DateInput(attrs={'type':'date', 'class':'form-control'})
    )
    report_type = forms.ChoiceField(
        choices=[('all','All'), ('revenue','Revenue'), ('profit','Profit'), ('cash','Cash')],
        required=False,
        widget=forms.Select(attrs={'class':'form-select'})
    )


class GoalForm(forms.ModelForm):
    goal_type = forms.ChoiceField(
        choices=FinancialGoal.GOAL_TYPES,
        required=True,
        widget=forms.Select(attrs={"class": "form-select"})
    )

    class Meta:
        model = FinancialGoal
        fields = ["goal_type", "target_value", "deadline"]
        widgets = {
            "target_value": forms.NumberInput(attrs={"class": "form-control", "placeholder": "Enter target amount"}),
            "deadline": forms.DateInput(attrs={"type": "date", "class": "form-control"}),
        }


class ProfileForm(forms.ModelForm):
    email = forms.EmailField(
        required=False,
        disabled=True,  # 🔒 makes it read-only
        widget=forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'Your email'}),
        label="Email"
    )

    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'email']  # ✅ added last_name
        widgets = {
            'first_name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'First name'}),
            'last_name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Last name'}),
        }
        labels = {
            'first_name': 'First Name',
            'last_name': 'Last Name',
        }

class PreferencesForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = [ 'email_notifications']

class PasswordChangeForm(forms.Form):
    current_password = forms.CharField(widget=forms.PasswordInput, required=True)
    new_password = forms.CharField(widget=forms.PasswordInput, required=True)
    confirm_password = forms.CharField(widget=forms.PasswordInput, required=True)




class AIQueryForm(forms.Form):
    query = forms.CharField(
        max_length=500,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Ask me about your finances...',
            'autocomplete': 'off'
        }),
        label=''
    )


class FinancialForecastForm(forms.Form):
    FORECAST_TYPES = [
        ('conservative', 'Conservative'),
        ('realistic', 'Realistic'),
        ('optimistic', 'Optimistic')
    ]
    
    forecast_period = forms.IntegerField(
        min_value=1, max_value=36,
        initial=12,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Months to forecast'
        })
    )
    
    forecast_type = forms.ChoiceField(
        choices=FORECAST_TYPES,
        initial='realistic',
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    market_conditions = forms.FloatField(
        required=False,
        initial=0.0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Market growth % (optional)',
            'step': '0.1'
        })
    )


