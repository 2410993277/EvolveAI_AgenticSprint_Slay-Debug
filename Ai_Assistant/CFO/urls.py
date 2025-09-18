from django.urls import path
from . import views

app_name = "CFO"

urlpatterns = [
    path('', views.index, name='index'),
    path('home/', views.index, name='index'),  # General home page

    # Authentication
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),

    # Reports & Dashboard
    path('upload/', views.upload_report, name='upload_report'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('report/<int:pk>/', views.report_detail, name='report_detail'),

    # Scenario Planning & Analysis
    path("scenario/", views.scenario_planning, name="scenario_planning"),
    path("financial-analysis/", views.financial_analysis, name="financial_analysis"),
    path('cash-flow/', views.cash_flow_view, name='cash_flow'),

    # Goals
    path("goals/", views.goal_tracking, name="goal_tracking"),

    # Export Center
    path("export-center/", views.export_center, name="export_center"),
    path("export/reports/csv/", views.export_reports_csv, name="export_reports_csv"),
    path("export/reports/excel/", views.export_reports_excel, name="export_reports_excel"),
    path("export/reports/pdf/", views.export_reports_pdf, name="export_reports_pdf"),

    # Reports Generator (if implemented)
    path("reports-generator/", views.reports_generator, name="reports_generator"),
    path("performance-metrics/", views.performance_metrics, name="performance_metrics"),
    path('settings/', views.settings_view, name='settings_view'),
    path('delete-account/', views.delete_account, name='delete_account'),
    path('help/', views.help_view, name='help_view'),
    path('contact/', views.contact_view, name='contact_view'),
    path('report/<int:pk>/download/', views.download_report_pdf, name='download_report_pdf'),
    path("ai-assistant/", views.ai_assistant, name="ai_assistant"),
    path("smart-insights/", views.smart_insights, name="smart_insights"),
    path("financial-health/", views.financial_health, name="financial_health"),
    path('risk-alerts/', views.risk_alerts, name='risk_alerts'),
    path("profile/", views.profile_view, name="profile")
]
