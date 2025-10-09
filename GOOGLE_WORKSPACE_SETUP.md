# Google Workspace Integration - Production Setup Guide

## Overview

The GRETA PAI system includes production-ready Google Workspace integration for Gmail, Calendar, and Drive APIs. The code is structured to work in mock mode during development and seamlessly switch to production mode when OAuth credentials are configured.

---

## Prerequisites

1. **Google Cloud Project**
   - Create a project at [Google Cloud Console](https://console.cloud.google.com/)
   - Enable billing (required for OAuth)

2. **Enable Required APIs**
   - Gmail API
   - Google Calendar API
   - Google Drive API

3. **OAuth 2.0 Credentials**
   - Create OAuth 2.0 Client ID (Web application type)
   - Configure authorized redirect URIs

---

## Required Environment Variables

Add these to your `.env` file:

```bash
# Google OAuth Configuration
GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-client-secret
GOOGLE_REDIRECT_URI=http://localhost:8000/api/oauth/callback/google

# Optional: Google API Configuration
GOOGLE_API_SCOPES=https://www.googleapis.com/auth/gmail.modify,https://www.googleapis.com/auth/calendar,https://www.googleapis.com/auth/drive

# Mock Mode (set to false for production)
GOOGLE_MOCK_MODE=true
```

---

## Setup Steps

### 1. Create Google Cloud Project

```bash
# Go to Google Cloud Console
https://console.cloud.google.com/

# Create new project
Project Name: GRETA PAI
Project ID: greta-pai-[random]
```

### 2. Enable APIs

```bash
# Navigate to APIs & Services > Library
# Enable the following APIs:
- Gmail API
- Google Calendar API  
- Google Drive API
```

### 3. Create OAuth 2.0 Credentials

```bash
# Navigate to APIs & Services > Credentials
# Click "Create Credentials" > "OAuth 2.0 Client ID"

Application type: Web application
Name: GRETA PAI OAuth Client

Authorized JavaScript origins:
- http://localhost:3000
- http://localhost:5173
- http://localhost:8000

Authorized redirect URIs:
- http://localhost:8000/api/oauth/callback/google
- http://localhost:8000/api/oauth/callback

# For production, add your domain:
- https://yourdomain.com/api/oauth/callback/google
```

### 4. Download Credentials

```bash
# After creating credentials, download the JSON file
# Extract the client_id and client_secret
# Add them to your .env file
```

### 5. Configure OAuth Scopes

The following scopes are required:

**Gmail:**
- `https://www.googleapis.com/auth/gmail.modify` - Read, compose, send, and permanently delete email
- `https://www.googleapis.com/auth/gmail.readonly` - Read email (alternative for read-only)

**Calendar:**
- `https://www.googleapis.com/auth/calendar` - Full calendar access
- `https://www.googleapis.com/auth/calendar.events` - Events only (alternative)

**Drive:**
- `https://www.googleapis.com/auth/drive` - Full Drive access
- `https://www.googleapis.com/auth/drive.file` - Files created by app only (alternative)

---

## Code Structure

### Services with Error Handling

All Google Workspace services include:

✅ **Retry Logic** - Exponential backoff with jitter
✅ **Rate Limiting** - Respects API quotas
✅ **Error Handling** - Custom exceptions for different error types
✅ **Token Refresh** - Automatic OAuth token refresh
✅ **Mock Mode** - Development without real credentials

### Files

```
backend/
├── services/
│   ├── gmail_service.py      # Gmail API integration
│   ├── calendar_service.py   # Calendar API integration
│   ├── drive_service.py      # Drive API integration
│   └── oauth_service.py      # OAuth token management
├── utils/
│   └── google_api_utils.py   # Error handling & retry logic
├── models/
│   ├── gmail_models.py       # Gmail data models
│   ├── calendar_models.py    # Calendar data models
│   ├── drive_models.py       # Drive data models
│   └── oauth_models.py       # OAuth data models
└── routers/
    ├── gmail_router.py       # Gmail API endpoints
    ├── calendar_router.py    # Calendar API endpoints
    ├── drive_router.py       # Drive API endpoints
    └── oauth_router.py       # OAuth flow endpoints
```

---

## Testing

### 1. Test OAuth Flow

```bash
# Start backend
cd backend
uvicorn main:app --reload

# Navigate to OAuth initiation
http://localhost:8000/api/oauth/authorize/google

# Complete OAuth flow
# You'll be redirected to Google login
# Grant permissions
# Redirected back to your app with tokens
```

### 2. Test Gmail API

```bash
# Send test email
curl -X POST http://localhost:8000/api/gmail/send \
  -H "Content-Type: application/json" \
  -d '{
    "to": "recipient@example.com",
    "subject": "Test Email",
    "body": "This is a test email from GRETA PAI"
  }'
```

### 3. Test Calendar API

```bash
# List calendars
curl http://localhost:8000/api/calendar/calendars

# Create event
curl -X POST http://localhost:8000/api/calendar/events \
  -H "Content-Type: application/json" \
  -d '{
    "summary": "Test Event",
    "start": "2025-10-10T10:00:00Z",
    "end": "2025-10-10T11:00:00Z"
  }'
```

### 4. Test Drive API

```bash
# List files
curl http://localhost:8000/api/drive/files

# Upload file
curl -X POST http://localhost:8000/api/drive/upload \
  -F "file=@test.pdf"
```

---

## Switching from Mock to Production Mode

### Current State (Mock Mode)

```python
# In backend/services/gmail_service.py
self.mock_mode = True  # Returns mock data
```

### Production Mode

1. **Set environment variable:**
   ```bash
   GOOGLE_MOCK_MODE=false
   ```

2. **Or update code:**
   ```python
   # In backend/services/gmail_service.py
   self.mock_mode = os.getenv("GOOGLE_MOCK_MODE", "true").lower() == "true"
   ```

3. **Restart backend:**
   ```bash
   uvicorn main:app --reload
   ```

---

## API Rate Limits

Google Workspace APIs have the following quotas:

**Gmail API:**
- 250 quota units per user per second
- 1,000,000,000 quota units per day

**Calendar API:**
- 1,000,000 queries per day
- 500 queries per 100 seconds per user

**Drive API:**
- 1,000 queries per 100 seconds per user
- 10,000 queries per 100 seconds

The code includes automatic rate limiting to stay within these quotas.

---

## Error Handling

The integration handles these error scenarios:

✅ **401 Unauthorized** - Token expired, automatic refresh
✅ **403 Forbidden** - Insufficient permissions, clear error message
✅ **404 Not Found** - Resource doesn't exist
✅ **429 Rate Limit** - Automatic retry with exponential backoff
✅ **500 Server Error** - Retry with backoff
✅ **Network Errors** - Retry with backoff

---

## Security Best Practices

1. **Never commit credentials** - Use `.env` file (in `.gitignore`)
2. **Use HTTPS in production** - Required for OAuth
3. **Rotate secrets regularly** - Update client secret periodically
4. **Limit scopes** - Only request necessary permissions
5. **Store tokens securely** - Encrypted in database
6. **Implement token refresh** - Automatic refresh before expiry
7. **Audit API calls** - Log all Google API interactions

---

## Troubleshooting

### "Invalid credentials" error

```bash
# Check environment variables
echo $GOOGLE_CLIENT_ID
echo $GOOGLE_CLIENT_SECRET

# Verify they match Google Cloud Console
```

### "Redirect URI mismatch" error

```bash
# Ensure redirect URI in code matches Google Cloud Console
# Check both:
# - Authorized redirect URIs in Google Cloud Console
# - GOOGLE_REDIRECT_URI in .env file
```

### "Insufficient permissions" error

```bash
# User needs to re-authorize with updated scopes
# Delete existing tokens and re-authenticate
```

### Rate limit errors

```bash
# The code automatically retries with exponential backoff
# If persistent, check Google Cloud Console quotas
```

---

## Production Deployment Checklist

- [ ] Google Cloud Project created
- [ ] APIs enabled (Gmail, Calendar, Drive)
- [ ] OAuth credentials created
- [ ] Environment variables configured
- [ ] Redirect URIs updated for production domain
- [ ] HTTPS enabled
- [ ] Token storage encrypted
- [ ] Rate limiting configured
- [ ] Error logging enabled
- [ ] Monitoring setup
- [ ] Backup OAuth credentials stored securely

---

## Next Steps

Once credentials are configured:

1. Set `GOOGLE_MOCK_MODE=false` in `.env`
2. Restart the backend
3. Test OAuth flow
4. Test each API (Gmail, Calendar, Drive)
5. Monitor logs for errors
6. Verify rate limiting works
7. Test token refresh

---

## Support

For issues with Google Workspace integration:

1. Check Google Cloud Console for API status
2. Review backend logs for detailed error messages
3. Verify environment variables are set correctly
4. Test OAuth flow in isolation
5. Check API quotas in Google Cloud Console

---

**Status**: ✅ Code is production-ready, waiting for OAuth credentials configuration
