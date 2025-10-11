# Backend API Integration Guide
## CPAS4/Greta - Complete Implementation Roadmap

**Created**: 2025-10-11  
**Status**: Ready for Implementation

---

## Overview

This guide provides step-by-step instructions for integrating the newly created backend routers (Voice, MCP, Hook) with the CPAS4 frontend UI.

---

## ✅ Completed Work

### Backend Routers Created

1. **`voice_router.py`** (555 lines)
   - Location: `/Users/macone/Desktop/Greta/backend/routers/voice_router.py`
   - Endpoints: 20+ endpoints for STT, TTS, VAD, voice commands, profiles, conversation mode
   - Prefix: `/api/v1/voice`
   - Status: ✅ Created, needs registration in main.py

2. **`hook_router.py`** (593 lines)
   - Location: `/Users/macone/Desktop/Greta/backend/routers/hook_router.py`
   - Endpoints: 25+ endpoints for hooks, plugins, middleware, events
   - Prefix: `/api/v1/hooks`
   - Status: ✅ Created, needs registration in main.py

3. **`mcp_router.py`** (170 lines)
   - Location: `/Users/macone/Desktop/Greta/backend/routers/mcp_router.py`
   - Endpoints: 30+ endpoints for MCP servers, tools, discovery, integrations
   - Prefix: `/api/v1/mcp`
   - Status: ✅ Already exists and registered

### Frontend API Clients Created

1. **`voiceApi.ts`** (300+ lines)
   - Location: `/Users/macone/Desktop/Greta/cpas4-ui/src/api/voiceApi.ts`
   - Functions: 20+ API client functions
   - Status: ✅ Created, ready to use

2. **`mcpApi.ts`** (300+ lines)
   - Location: `/Users/macone/Desktop/Greta/cpas4-ui/src/api/mcpApi.ts`
   - Functions: 20+ API client functions
   - Status: ✅ Created, ready to use

3. **`hookApi.ts`** (300+ lines)
   - Location: `/Users/macone/Desktop/Greta/cpas4-ui/src/api/hookApi.ts`
   - Functions: 20+ API client functions
   - Status: ✅ Created, ready to use

---

## 🔧 Step 1: Register Backend Routers

### 1.1 Register Voice Router

**File**: `/Users/macone/Desktop/Greta/backend/main.py`

**Action**: Add voice router registration

```python
# Around line 400-450 (where other routers are registered)

# Force include Voice Router
if force_include and router_name == 'voice_router':
    try:
        from routers.voice_router import router as voice_router
        app.include_router(voice_router, tags=["Voice"])
        routers_loaded += 1
        logger.info(f"✅ Voice Router forced inclusion (no additional prefix)")
        continue
    except Exception as e:
        logger.warning(f"⚠️ Voice router import failed: {e}")
```

**Also add to router list** (around line 250-300):

```python
AVAILABLE_ROUTERS = [
    # ... existing routers ...
    'voice_router',
    # ... rest of routers ...
]
```

### 1.2 Register Hook Router

**File**: `/Users/macone/Desktop/Greta/backend/main.py`

**Action**: Add hook router registration

```python
# Force include Hook Router
if force_include and router_name == 'hook_router':
    try:
        from routers.hook_router import router as hook_router
        app.include_router(hook_router, tags=["Hooks"])
        routers_loaded += 1
        logger.info(f"✅ Hook Router forced inclusion (no additional prefix)")
        continue
    except Exception as e:
        logger.warning(f"⚠️ Hook router import failed: {e}")
```

**Also add to router list**:

```python
AVAILABLE_ROUTERS = [
    # ... existing routers ...
    'hook_router',
    # ... rest of routers ...
]
```

### 1.3 Verify MCP Router

**File**: `/Users/macone/Desktop/Greta/backend/main.py`

**Action**: Verify `mcp_router` is already registered (it should be)

Search for `mcp_router` or `mcp_hub_router` in main.py to confirm.

---

## 🎨 Step 2: Update Frontend Components

### 2.1 Update VoiceControl.tsx

**File**: `/Users/macone/Desktop/Greta/cpas4-ui/src/components/voice/VoiceControl.tsx`

**Changes Needed**:

1. **Import API client** (already done):
   ```typescript
   import * as voiceApi from '../../api/voiceApi';
   ```

2. **Update `startListening` function** (lines 55-82):
   ```typescript
   const startListening = async () => {
     try {
       const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
       
       // Create MediaRecorder
       const mediaRecorder = new MediaRecorder(stream);
       const audioChunks: Blob[] = [];
       
       mediaRecorder.ondataavailable = (event) => {
         audioChunks.push(event.data);
       };
       
       mediaRecorder.onstop = async () => {
         const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
         const audioFile = new File([audioBlob], 'recording.wav', { type: 'audio/wav' });
         
         try {
           // Call backend STT API
           const result = await voiceApi.speechToText(audioFile);
           setTranscript(result.text);
           
           // Check for wake word
           const wakeWordResult = await voiceApi.detectWakeWord(result.text);
           if (wakeWordResult.has_wake_word) {
             toast.success('Wake word detected!');
             const command = await voiceApi.parseVoiceCommand(result.text);
             if (command) {
               toast.info(`Command: ${command.command}`);
               await voiceApi.executeVoiceCommand(result.text);
             }
           }
           
           if (onVoiceInput) {
             onVoiceInput(result.text);
           }
         } catch (error) {
           console.error('STT error:', error);
           toast.error('Speech recognition failed');
         }
       };
       
       mediaRecorder.start();
       
       // Visualization code (keep existing)
       if (audioContextRef.current && analyserRef.current) {
         const source = audioContextRef.current.createMediaStreamSource(stream);
         source.connect(analyserRef.current);
         visualizeAudio();
       }
       
       setIsListening(true);
       toast.success('Voice input started');
       
       // Auto-stop after 10 seconds
       setTimeout(() => {
         if (mediaRecorder.state === 'recording') {
           mediaRecorder.stop();
           stopListening();
         }
       }, 10000);
       
     } catch (error) {
       console.error('Error accessing microphone:', error);
       toast.error('Failed to access microphone');
     }
   };
   ```

3. **Update `speak` function** (lines 107-116):
   ```typescript
   const speak = async (text: string) => {
     try {
       setIsSpeaking(true);
       
       // Call backend TTS API
       const audioBlob = await voiceApi.textToSpeech({ text });
       
       // Play audio
       const audioUrl = URL.createObjectURL(audioBlob);
       const audio = new Audio(audioUrl);
       audio.onended = () => {
         setIsSpeaking(false);
         URL.revokeObjectURL(audioUrl);
       };
       audio.play();
       
     } catch (error) {
       console.error('TTS error:', error);
       toast.error('Text-to-speech failed');
       setIsSpeaking(false);
     }
   };
   ```

### 2.2 Update VoiceSettings.tsx

**File**: `/Users/macone/Desktop/Greta/cpas4-ui/src/components/voice/VoiceSettings.tsx`

**Changes Needed**:

1. **Add API imports**:
   ```typescript
   import * as voiceApi from '../../api/voiceApi';
   import { useEffect } from 'react';
   ```

2. **Load profiles on mount**:
   ```typescript
   useEffect(() => {
     loadProfiles();
     loadLanguages();
     loadVoices();
   }, []);
   
   const loadProfiles = async () => {
     try {
       const profiles = await voiceApi.listVoiceProfiles();
       setProfiles(profiles);
     } catch (error) {
       console.error('Failed to load profiles:', error);
       toast.error('Failed to load voice profiles');
     }
   };
   
   const loadLanguages = async () => {
     try {
       const langs = await voiceApi.getSupportedLanguages();
       // Update languages state
     } catch (error) {
       console.error('Failed to load languages:', error);
     }
   };
   
   const loadVoices = async () => {
     try {
       const voices = await voiceApi.getSupportedVoices(ttsEngine);
       // Update voices state
     } catch (error) {
       console.error('Failed to load voices:', error);
     }
   };
   ```

3. **Update save settings**:
   ```typescript
   const handleSaveSettings = async () => {
     try {
       // Save to backend
       await voiceApi.updateVoiceProfile('default', {
         preferred_language: selectedLanguage,
         preferred_voice: selectedVoice,
         settings: {
           stt_engine: sttEngine,
           tts_engine: ttsEngine,
         },
       });
       
       toast.success('Voice settings saved');
     } catch (error) {
       console.error('Failed to save settings:', error);
       toast.error('Failed to save settings');
     }
   };
   ```

### 2.3 Update MCPManager.tsx

**File**: `/Users/macone/Desktop/Greta/cpas4-ui/src/components/mcp/MCPManager.tsx`

**Changes Needed**:

1. **Add API imports**:
   ```typescript
   import * as mcpApi from '../../api/mcpApi';
   import { useEffect } from 'react';
   ```

2. **Load servers and tools on mount**:
   ```typescript
   useEffect(() => {
     loadServers();
     loadTools();
   }, []);
   
   const loadServers = async () => {
     try {
       const servers = await mcpApi.listServers();
       setServers(servers);
     } catch (error) {
       console.error('Failed to load servers:', error);
       toast.error('Failed to load MCP servers');
     }
   };
   
   const loadTools = async () => {
     try {
       const tools = await mcpApi.listTools();
       setTools(tools);
     } catch (error) {
       console.error('Failed to load tools:', error);
       toast.error('Failed to load MCP tools');
     }
   };
   ```

3. **Implement server actions**:
   ```typescript
   const handleAddServer = async (server: { name: string; url: string }) => {
     try {
       await mcpApi.createServer(server);
       await loadServers();
       toast.success('Server added successfully');
     } catch (error) {
       console.error('Failed to add server:', error);
       toast.error('Failed to add server');
     }
   };
   
   const handleTestConnection = async (serverId: string) => {
     try {
       const result = await mcpApi.testServerConnection(serverId);
       if (result.success) {
         toast.success('Connection successful');
       } else {
         toast.error('Connection failed');
       }
     } catch (error) {
       console.error('Connection test failed:', error);
       toast.error('Connection test failed');
     }
   };
   
   const handleToggleTool = async (toolId: string, enabled: boolean) => {
     try {
       await mcpApi.toggleTool(toolId, enabled);
       await loadTools();
       toast.success(`Tool ${enabled ? 'enabled' : 'disabled'}`);
     } catch (error) {
       console.error('Failed to toggle tool:', error);
       toast.error('Failed to toggle tool');
     }
   };
   ```

### 2.4 Update HookManager.tsx

**File**: `/Users/macone/Desktop/Greta/cpas4-ui/src/components/hooks/HookManager.tsx`

**Changes Needed**:

1. **Add API imports**:
   ```typescript
   import * as hookApi from '../../api/hookApi';
   import { useEffect } from 'react';
   ```

2. **Load hooks and plugins on mount**:
   ```typescript
   useEffect(() => {
     loadHooks();
     loadPlugins();
     loadEventTypes();
   }, []);
   
   const loadHooks = async () => {
     try {
       const hooks = await hookApi.listHooks();
       setHooks(hooks);
     } catch (error) {
       console.error('Failed to load hooks:', error);
       toast.error('Failed to load hooks');
     }
   };
   
   const loadPlugins = async () => {
     try {
       const plugins = await hookApi.listPlugins();
       setPlugins(plugins);
     } catch (error) {
       console.error('Failed to load plugins:', error);
       toast.error('Failed to load plugins');
     }
   };
   
   const loadEventTypes = async () => {
     try {
       const events = await hookApi.getEventTypes();
       setEventTypes(events);
     } catch (error) {
       console.error('Failed to load event types:', error);
     }
   };
   ```

3. **Implement hook actions**:
   ```typescript
   const handleAddHook = async (hook: { hook_id: string; event_type: string; hook_type: string; priority: number }) => {
     try {
       await hookApi.registerHook(hook);
       await loadHooks();
       toast.success('Hook registered successfully');
     } catch (error) {
       console.error('Failed to register hook:', error);
       toast.error('Failed to register hook');
     }
   };
   
   const handleTestHook = async (hookId: string) => {
     try {
       const result = await hookApi.testHook(hookId, { test: 'data' });
       if (result.success) {
         toast.success('Hook test successful');
       } else {
         toast.error(`Hook test failed: ${result.error}`);
       }
     } catch (error) {
       console.error('Hook test failed:', error);
       toast.error('Hook test failed');
     }
   };
   
   const handleTogglePlugin = async (pluginId: string, enabled: boolean) => {
     try {
       if (enabled) {
         await hookApi.enablePlugin(pluginId);
       } else {
         await hookApi.disablePlugin(pluginId);
       }
       await loadPlugins();
       toast.success(`Plugin ${enabled ? 'enabled' : 'disabled'}`);
     } catch (error) {
       console.error('Failed to toggle plugin:', error);
       toast.error('Failed to toggle plugin');
     }
   };
   ```

---

## 🧪 Step 3: Testing

### 3.1 Backend Testing

1. **Start backend server**:
   ```bash
   cd /Users/macone/Desktop/Greta/backend
   python main.py
   ```

2. **Verify routers are loaded**:
   - Check console output for "✅ Voice Router forced inclusion"
   - Check console output for "✅ Hook Router forced inclusion"
   - Check console output for "✅ MCP Router" (should already exist)

3. **Test endpoints**:
   ```bash
   # Test Voice API
   curl http://localhost:8000/api/v1/voice/languages
   curl http://localhost:8000/api/v1/voice/voices
   curl http://localhost:8000/api/v1/voice/stats
   
   # Test Hook API
   curl http://localhost:8000/api/v1/hooks/events
   curl http://localhost:8000/api/v1/hooks/health
   curl http://localhost:8000/api/v1/hooks/stats
   
   # Test MCP API
   curl http://localhost:8000/api/v1/mcp/health
   curl http://localhost:8000/api/v1/mcp/stats
   curl http://localhost:8000/api/v1/mcp/servers
   ```

### 3.2 Frontend Testing

1. **Start frontend dev server**:
   ```bash
   cd /Users/macone/Desktop/Greta/cpas4-ui
   npm run dev
   ```

2. **Navigate to pages**:
   - http://localhost:5173/voice
   - http://localhost:5173/mcp
   - http://localhost:5173/hooks

3. **Test features**:
   - Voice Control: Click microphone, speak, verify transcription
   - MCP Manager: Add server, test connection, toggle tools
   - Hook Manager: Add hook, test hook, toggle plugins

---

## 🐛 Common Issues & Solutions

### Issue 1: CORS Errors

**Symptom**: Browser console shows CORS errors when calling API

**Solution**: Verify CORS middleware in backend `main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Issue 2: 503 Service Unavailable

**Symptom**: API returns 503 errors

**Solution**: Voice/Hook services may not be initialized. Check:
1. Services are imported correctly in routers
2. Service instances are created
3. Dependencies are installed

### Issue 3: Microphone Access Denied

**Symptom**: Browser doesn't allow microphone access

**Solution**: 
1. Use HTTPS or localhost
2. Grant microphone permissions in browser settings
3. Check browser console for specific errors

---

## 📊 Progress Tracking

- [x] Create voice_router.py
- [x] Create hook_router.py
- [x] Verify mcp_router.py exists
- [x] Create voiceApi.ts
- [x] Create mcpApi.ts
- [x] Create hookApi.ts
- [ ] Register voice_router in main.py
- [ ] Register hook_router in main.py
- [ ] Update VoiceControl.tsx
- [ ] Update VoiceSettings.tsx
- [ ] Update MCPManager.tsx
- [ ] Update HookManager.tsx
- [ ] Test backend endpoints
- [ ] Test frontend integration
- [ ] Fix navigation issues
- [ ] Deploy to production

---

## 🚀 Next Steps

After completing this integration:

1. **Request 2**: Fix UI Navigation (see NAVIGATION_FIX_GUIDE.md)
2. **Request 3**: Comprehensive Codebase Analysis (see CODEBASE_ANALYSIS.md)
3. **Request 4**: Desktop Application Conversion (see DESKTOP_APP_GUIDE.md)

---

**End of Backend Integration Guide**

