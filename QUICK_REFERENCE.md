# CPAS4/Greta Quick Reference Card
## Fast Access to All Implementation Guides

**Last Updated**: 2025-10-11

---

## 📚 Documentation Index

| Guide | Purpose | Lines | Time to Read |
|-------|---------|-------|--------------|
| [IMPLEMENTATION_SUMMARY.md](#) | Overview of all work | 300 | 10 min |
| [BACKEND_INTEGRATION_GUIDE.md](#) | Connect UI to backend | 300 | 15 min |
| [NAVIGATION_VERIFICATION.md](#) | Navigation audit | 300 | 10 min |
| [CODEBASE_ANALYSIS.md](#) | Issues & solutions | 300 | 20 min |
| [DESKTOP_APP_GUIDE.md](#) | Tauri desktop app | 300 | 20 min |

---

## 🚀 Quick Commands

### Start Development

```bash
# Backend
cd /Users/macone/Desktop/Greta/backend
python main.py

# Frontend
cd /Users/macone/Desktop/Greta/cpas4-ui
npm run dev

# Both running:
# Backend: http://localhost:8000
# Frontend: http://localhost:5173
```

### Test API Endpoints

```bash
# Voice API
curl http://localhost:8000/api/v1/voice/languages
curl http://localhost:8000/api/v1/voice/stats

# Hook API
curl http://localhost:8000/api/v1/hooks/events
curl http://localhost:8000/api/v1/hooks/health

# MCP API
curl http://localhost:8000/api/v1/mcp/servers
curl http://localhost:8000/api/v1/mcp/stats
```

### Build Desktop App

```bash
cd /Users/macone/Desktop/Greta/cpas4-ui

# Development
npm run tauri dev

# Production
npm run tauri build

# Universal (Apple Silicon + Intel)
npm run tauri build -- --target universal-apple-darwin
```

---

## 📁 File Locations

### Backend Routers
```
/Users/macone/Desktop/Greta/backend/routers/
├── voice_router.py      (555 lines) ✅ Created
├── hook_router.py       (593 lines) ✅ Created
└── mcp_router.py        (170 lines) ✅ Exists
```

### Frontend API Clients
```
/Users/macone/Desktop/Greta/cpas4-ui/src/api/
├── voiceApi.ts          (300+ lines) ✅ Created
├── mcpApi.ts            (300+ lines) ✅ Created
└── hookApi.ts           (300+ lines) ✅ Created
```

### UI Components
```
/Users/macone/Desktop/Greta/cpas4-ui/src/components/
├── voice/
│   ├── VoiceControl.tsx     (240 lines) ✅ Exists
│   └── VoiceSettings.tsx    (210 lines) ✅ Exists
├── mcp/
│   └── MCPManager.tsx       (230 lines) ✅ Exists
└── hooks/
    └── HookManager.tsx      (280 lines) ✅ Exists
```

### Documentation
```
/Users/macone/Desktop/Greta/
├── IMPLEMENTATION_SUMMARY.md        ✅ Created
├── BACKEND_INTEGRATION_GUIDE.md     ✅ Created
├── NAVIGATION_VERIFICATION.md       ✅ Created
├── CODEBASE_ANALYSIS.md             ✅ Created
├── DESKTOP_APP_GUIDE.md             ✅ Created
└── QUICK_REFERENCE.md               ✅ This file
```

---

## 🎯 Implementation Checklist

### Phase 1: Backend Integration (12 hours)
- [ ] Register voice_router in main.py
- [ ] Register hook_router in main.py
- [ ] Initialize services in lifespan
- [ ] Test all endpoints
- [ ] Update VoiceControl.tsx
- [ ] Update VoiceSettings.tsx
- [ ] Update MCPManager.tsx
- [ ] Update HookManager.tsx

### Phase 2: Critical Fixes (20 hours)
- [ ] Service initialization
- [ ] CORS configuration
- [ ] JWT authentication
- [ ] Database connection pooling

### Phase 3: High Priority (40 hours)
- [ ] Error handling middleware
- [ ] Rate limiting
- [ ] State management
- [ ] Loading states

### Phase 4: Desktop App (20 hours)
- [ ] Install Tauri
- [ ] Initialize Tauri
- [ ] Implement Rust backend
- [ ] Update frontend
- [ ] Build and test

### Phase 5: Deploy (15 hours)
- [ ] Code signing
- [ ] Auto-updates
- [ ] Distribution
- [ ] Production deploy

**Total**: 107 hours

---

## 🔧 Common Tasks

### Add New API Endpoint

**Backend** (`backend/routers/voice_router.py`):
```python
@router.get("/new-endpoint")
async def new_endpoint():
    return {"message": "Hello"}
```

**Frontend** (`cpas4-ui/src/api/voiceApi.ts`):
```typescript
export async function newEndpoint(): Promise<any> {
  const response = await fetch(`${API_BASE}/new-endpoint`);
  return response.json();
}
```

**Component** (`cpas4-ui/src/components/...`):
```typescript
import * as voiceApi from '../../api/voiceApi';

const handleClick = async () => {
  const result = await voiceApi.newEndpoint();
  console.log(result);
};
```

### Add New UI Component

```bash
cd /Users/macone/Desktop/Greta/cpas4-ui/src/components

# Create new component
mkdir my-feature
touch my-feature/MyComponent.tsx

# Add to routes (App.tsx)
import { MyComponent } from './components/my-feature/MyComponent';

<Route path="/my-feature" element={<MyComponent />} />

# Add to navigation (LeftSidebar.tsx)
<button onClick={() => navigate('/my-feature')}>
  <Icon className="h-4 w-4" />
  <span>My Feature</span>
</button>
```

### Debug Backend

```bash
# View logs
tail -f /Users/macone/Desktop/Greta/backend/logs/app.log

# Check service status
curl http://localhost:8000/health

# Test specific endpoint
curl -X POST http://localhost:8000/api/v1/voice/stt \
  -F "audio=@test.wav" \
  -F "language=en"
```

### Debug Frontend

```bash
# Check browser console
# Open DevTools: Cmd+Option+I

# Check network requests
# DevTools > Network tab

# Check React components
# Install React DevTools extension
```

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CPAS4/Greta System                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │   Frontend UI    │         │   Backend API    │         │
│  │  (React/Vite)    │◄───────►│   (FastAPI)      │         │
│  │  localhost:5173  │  HTTP   │  localhost:8000  │         │
│  └──────────────────┘         └──────────────────┘         │
│         │                              │                     │
│         │                              │                     │
│  ┌──────▼──────────┐         ┌────────▼─────────┐          │
│  │  UI Components  │         │    Services      │          │
│  │  - Voice        │         │  - Voice Service │          │
│  │  - MCP          │         │  - Hook Service  │          │
│  │  - Hooks        │         │  - MCP Service   │          │
│  └─────────────────┘         │  - LLM Service   │          │
│                               │  - RAG Service   │          │
│                               │  - etc.          │          │
│                               └──────────────────┘          │
│                                       │                      │
│                               ┌───────▼──────────┐          │
│                               │    Databases     │          │
│                               │  - MongoDB       │          │
│                               │  - ChromaDB      │          │
│                               └──────────────────┘          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🐛 Troubleshooting

### Backend won't start
```bash
# Check Python version
python --version  # Should be 3.13

# Check dependencies
pip list | grep fastapi

# Check port
lsof -i :8000
```

### Frontend won't start
```bash
# Check Node version
node --version  # Should be 18+

# Clear cache
rm -rf node_modules
npm install

# Check port
lsof -i :5173
```

### API calls fail
```bash
# Check CORS
# Verify backend allows http://localhost:5173

# Check backend is running
curl http://localhost:8000/health

# Check network tab in browser DevTools
```

### Desktop app won't build
```bash
# Check Rust
rustc --version

# Check Tauri
tauri --version

# Clean build
cd src-tauri
cargo clean
cd ..
npm run tauri build
```

---

## 📞 Support Resources

### Documentation
- FastAPI: https://fastapi.tiangolo.com/
- React: https://react.dev/
- Tauri: https://tauri.app/
- Vite: https://vitejs.dev/

### Community
- FastAPI Discord: https://discord.gg/fastapi
- React Discord: https://discord.gg/react
- Tauri Discord: https://discord.gg/tauri

### Tools
- API Testing: Postman, Insomnia
- Database: MongoDB Compass
- Debugging: Chrome DevTools, VS Code

---

## 🎓 Learning Resources

### Backend Development
- FastAPI Tutorial: https://fastapi.tiangolo.com/tutorial/
- Python Async: https://realpython.com/async-io-python/
- MongoDB: https://www.mongodb.com/docs/

### Frontend Development
- React Tutorial: https://react.dev/learn
- TypeScript: https://www.typescriptlang.org/docs/
- Tailwind CSS: https://tailwindcss.com/docs

### Desktop Development
- Tauri Guide: https://tauri.app/v1/guides/
- Rust Book: https://doc.rust-lang.org/book/

---

## 📈 Performance Benchmarks

### Current Performance
- Backend startup: ~2-3 seconds
- Frontend load: ~1-2 seconds
- API response time: ~50-200ms
- Memory usage: ~500 MB (backend + frontend)

### Target Performance (after optimization)
- Backend startup: ~1 second
- Frontend load: <1 second
- API response time: ~20-50ms
- Memory usage: ~300 MB

---

## 🔐 Security Checklist

- [ ] JWT authentication implemented
- [ ] API rate limiting enabled
- [ ] Input validation on all endpoints
- [ ] CORS properly configured
- [ ] HTTPS in production
- [ ] Secrets in environment variables
- [ ] SQL injection prevention
- [ ] XSS protection
- [ ] CSRF protection
- [ ] Security headers

---

## 🚀 Deployment Checklist

### Backend
- [ ] Environment variables configured
- [ ] Database migrations run
- [ ] SSL certificates installed
- [ ] Monitoring set up
- [ ] Logging configured
- [ ] Backup strategy in place

### Frontend
- [ ] Build optimized
- [ ] Assets compressed
- [ ] CDN configured
- [ ] Analytics added
- [ ] Error tracking enabled

### Desktop App
- [ ] Code signed
- [ ] Notarized (macOS)
- [ ] Auto-updates configured
- [ ] Crash reporting enabled
- [ ] Distribution packages created

---

## 📝 Notes

- All guides are in `/Users/macone/Desktop/Greta/`
- Backend code is in `/Users/macone/Desktop/Greta/backend/`
- Frontend code is in `/Users/macone/Desktop/Greta/cpas4-ui/`
- All routers use prefix `/api/v1/`
- All API clients use `http://localhost:8000` as base URL

---

**For detailed information, refer to the specific guides listed at the top of this document.**

**End of Quick Reference**

