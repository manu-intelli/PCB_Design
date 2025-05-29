import os
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv
import dj_database_url

# Load environment variables from .env file
load_dotenv()

# Base directory for the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Secret key used for Django security (Ensure this is set in the .env file)
SECRET_KEY = os.getenv('SECRET_KEY', 'default-secret-key')

# Environment setting (either 'LOCAL' or 'LIVE')
SERVER_TYPE = os.getenv('SERVER_TYPE', 'LOCAL').upper()

SERVER_NAME = os.getenv('SERVER_NAME',"LOCAL")

# Debug mode: Set to True for local development
DEBUG = True

# Frontend IP and Port configuration (used for CORS and API requests)
FRONTEND_IP = os.getenv("FRONTEND_IP", "http://localhost")
FRONTEND_PORT = os.getenv("FRONTEND_PORT", "5173")

# Configure allowed hosts based on the environment
if SERVER_TYPE == 'LOCAL':
    # Only allow localhost for local development
    ALLOWED_HOSTS = ['localhost', '127.0.0.1']
else:
    # For live servers, allow the production URL and frontend IP
    ALLOWED_HOSTS = ['pcb-design-5nqf.onrender.com','localhost', os.getenv('FRONTEND_IP', '*'),os.getenv('BACKEND_IP', '*')]

# List of installed apps required for the project
INSTALLED_APPS = [
    'django.contrib.admin',  # Django admin
    'django.contrib.auth',   # Authentication framework
    'django.contrib.contenttypes',  # Content types framework
    'django.contrib.sessions',  # Session management
    'django.contrib.messages',  # Messaging framework
    'django.contrib.staticfiles',  # Static file handling
    'rest_framework',  # Django REST framework for building APIs
    'rest_framework_simplejwt',  # JWT authentication
    'import_export',  # Import/Export functionality
    'drf_yasg',  # Swagger/OpenAPI documentation
    'corsheaders',  # Cross-origin resource sharing
    'right_to_draw',  # Custom app (your app)
    'authentication',  # Custom authentication app
    'masters',  # Custom app for managing master data
    'pibase',  # Custom app for managing PiBase records
]

# Middleware configuration
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',  # Security middleware
    # 'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',  # Session handling
    'corsheaders.middleware.CorsMiddleware',  # CORS middleware for cross-origin requests
    'django.middleware.common.CommonMiddleware',  # Common middleware
    'django.middleware.csrf.CsrfViewMiddleware',  # CSRF protection
    'django.contrib.auth.middleware.AuthenticationMiddleware',  # Authentication middleware
    'middleware.current_user_middleware.CurrentUserMiddleware',  # Custom middleware for user management
    'django.contrib.messages.middleware.MessageMiddleware',  # Message framework
    'django.middleware.clickjacking.XFrameOptionsMiddleware',  # Clickjacking protection
]

# STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'
# Root URL configuration (links to the main URL routing file)
ROOT_URLCONF = 'pcb_design.urls'

# Template settings for rendering HTML views (if any)
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],  # Add your template directories here if required
        'APP_DIRS': True,  # Enable app-specific template directories
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

# WSGI application for deployment
WSGI_APPLICATION = 'pcb_design.wsgi.application'

# Custom user model for authentication (set to the model you defined)
AUTH_USER_MODEL = 'authentication.CustomUser'

# Database configuration
ODBC_DRIVER = 'ODBC Driver 17 for SQL Server' if SERVER_TYPE == 'LOCAL' else 'ODBC Driver 13 for SQL Server'

DATABASES = {
    'default': {
        'ENGINE': 'mssql',  # Use MS SQL database engine
        'NAME': os.getenv('DB_NAME'),  # Database name from .env
        'USER': os.getenv('DB_USER'),  # Database user from .env
        'PASSWORD': os.getenv('DB_PASSWORD'),  # Database password from .env
        'HOST': os.getenv('DB_HOST'),  # Database host from .env
        'PORT': os.getenv('DB_PORT'),  # Database port from .env
        'OPTIONS': {
            'driver': ODBC_DRIVER,  # Choose the appropriate ODBC driver
            'autocommit': True,  # Enable autocommit for transactions
            'extra_params': 'DataTypeCompatibility=80;MARS Connection=True;',  # MS SQL options
            'use_legacy_date_fields': True,  # Legacy date fields support
        },
    }
}

# Password validation settings
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# REST Framework configuration
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'authentication.custom_authentication.CustomJWTAuthentication',  # Custom JWT authentication class
    ),
    'DEFAULT_PERMISSION_CLASSES': (
        'authentication.custom_permissions.IsAuthorized',  # Custom permission class
    ),
}

# JWT token expiration settings
SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(minutes=150),  # Access token lifetime
    'REFRESH_TOKEN_LIFETIME': timedelta(days=1),  # Refresh token lifetime
}

# Localization and timezone settings
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True  # Enable internationalization
USE_TZ = True  # Enable timezone support

# Static file settings
STATIC_URL = '/static/'  # URL path for static files
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')  # Directory to store static files

MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')


# Default auto field type for primary keys
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# URL append slash setting
APPEND_SLASH = False

# CORS settings to handle cross-origin requests
if SERVER_TYPE == 'LOCAL':
    CORS_ALLOWED_ORIGINS = [
        'http://127.0.0.1:5173',  # Allow local development frontend
        'http://localhost:5173',  # Allow localhost development frontend
        f"{FRONTEND_IP}:{FRONTEND_PORT}",
    ]
else:
    CORS_ALLOWED_ORIGINS = [
        f"{FRONTEND_IP}:{FRONTEND_PORT}",  # Allow frontend configured in .env for production
         'http://localhost:5173' if SERVER_NAME == 'DEV' else ''
    ]

# Allow credentials in CORS requests
CORS_ALLOW_CREDENTIALS = True

