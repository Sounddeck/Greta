# CPAS4 UI Visual Guide 🎨

**Updated**: 2025-10-11  
**Frontend**: http://localhost:5173

---

## UI Layout Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              HEADER (Top Bar)                                │
│  [Logo] CPAS4                                    [Theme] [Settings] [User]  │
├──────────────┬──────────────────────────────────────────────┬───────────────┤
│              │                                              │               │
│  LEFT        │           MAIN CONTENT AREA                  │    RIGHT      │
│  SIDEBAR     │                                              │   SIDEBAR     │
│              │                                              │               │
│ Agent Types  │  ┌────────────────────────────────────────┐  │ Voice Control │
│ - Research   │  │                                        │  │ - Microphone  │
│ - Coding     │  │                                        │  │ - Speaker     │
│ - Data...    │  │        Current Page Content            │  │ - Convers...  │
│ - Content... │  │                                        │  │               │
│ - Task...    │  │                                        │  │ Task History  │
│ - Customer.. │  │                                        │  │ - No tasks    │
│ - Sales      │  │                                        │  │               │
│ - Marketing  │  │                                        │  │ Active Jobs   │
│ - Finance    │  │                                        │  │ - No jobs     │
│ - General    │  │                                        │  │               │
│              │  │                                        │  │ System Status │
│ Quick Actions│  │                                        │  │ - Backend ✓   │
│ - New Agent  │  │                                        │  │ - Database ✓  │
│ - Execute... │  │                                        │  │ - Ollama ✓    │
│ - Generate.. │  │                                        │  │               │
│ - Agent...   │  └────────────────────────────────────────┘  │               │
│              │                                              │               │
│ Advanced ⭐  │                                              │               │
│ - Voice 🎤   │                                              │               │
│ - MCP 🖥️     │                                              │               │
│ - Hooks 🔗   │                                              │               │
│              │                                              │               │
│ Recent       │                                              │               │
│ - No recent  │                                              │               │
│              │                                              │               │
└──────────────┴──────────────────────────────────────────────┴───────────────┘
```

---

## Voice Control Page (`/voice`)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Voice Settings                                      │
│  Configure voice input and output preferences                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  🌍 Language                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ English                                                            ▼   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  🔊 Voice                                                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Default                                                            ▼   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  🎤 Speech-to-Text Engine                                                   │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Whisper (Local)                                                    ▼   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  🔊 Text-to-Speech Engine                                                   │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Coqui TTS (Local)                                                  ▼   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  👤 Voice Profiles                                    [Create Profile]      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Default Profile                                              [Edit]    │ │
│  │ English • Default                                                      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                          💾 Save Settings                              │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ℹ️ Voice Integration Features:                                             │
│  • Multi-language support (7 languages)                                     │
│  • Multiple STT/TTS engines                                                 │
│  • Voice profiles for personalization                                       │
│  • Wake word detection                                                      │
│  • Conversation mode                                                        │
│  • Audio format support (WAV, MP3, OGG, FLAC, RAW)                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Voice Control Widget (Right Sidebar)

```
┌─────────────────────────────────────────────────────────────┐
│  Voice Control                                    ⚙️        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Audio Level: ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 35%    │
│                                                              │
│  Transcript:                                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Hey Greta, generate a story about cats               │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────┐  ┌──────────────────────────┐    │
│  │ 🎤 Stop Listening    │  │ 🔊 Test Voice            │    │
│  └──────────────────────┘  └──────────────────────────┘    │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 👤 Start Conversation Mode                           │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  Wake Words:                                                 │
│  • "Hey Greta"                                               │
│  • "Greta"                                                   │
│  • "OK Greta"                                                │
│                                                              │
│  Commands:                                                   │
│  • "Generate [content]"                                      │
│  • "Search for [query]"                                      │
│  • "Create [item]"                                           │
│  • "Export [data]"                                           │
│  • "Stop" / "Help"                                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## MCP Manager Page (`/mcp`)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MCP Manager                                         │
│  Manage Model Context Protocol servers and tools                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  🖥️ MCP Servers                                        [+ Add Server]       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Greta MCP Server                                    [connected]        │ │
│  │ http://localhost:8000/mcp                                              │ │
│  │ 6 tools available                          [Test]  [🗑️]                │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  🔧 Available Tools                                                          │
│  ┌──────────────────────────────┐  ┌──────────────────────────────┐        │
│  │ Generate Text                │  │ Search Knowledge             │        │
│  │ Generate text using LLM      │  │ Search RAG knowledge base    │        │
│  │ Server: Greta MCP Server     │  │ Server: Greta MCP Server     │        │
│  │                   [Enabled]  │  │                   [Enabled]  │        │
│  └──────────────────────────────┘  └──────────────────────────────┘        │
│  ┌──────────────────────────────┐  ┌──────────────────────────────┐        │
│  │ Create Agent                 │  │ Export Data                  │        │
│  │ Create new agent from...     │  │ Export data to file          │        │
│  │ Server: Greta MCP Server     │  │ Server: Greta MCP Server     │        │
│  │                   [Enabled]  │  │                   [Enabled]  │        │
│  └──────────────────────────────┘  └──────────────────────────────┘        │
│  ┌──────────────────────────────┐  ┌──────────────────────────────┐        │
│  │ Get Context                  │  │ Execute Hook                 │        │
│  │ Retrieve context info        │  │ Execute registered hook      │        │
│  │ Server: Greta MCP Server     │  │ Server: Greta MCP Server     │        │
│  │                   [Enabled]  │  │                   [Enabled]  │        │
│  └──────────────────────────────┘  └──────────────────────────────┘        │
│                                                                              │
│  🔑 API Keys                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Manage API keys for MCP authentication                                 │ │
│  │                                                                        │ │
│  │ [+ Generate API Key]                                                   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ℹ️ MCP Integration Features:                                               │
│  • Expose Greta tools via MCP protocol                                      │
│  • Connect to external MCP servers                                          │
│  • Tool registry with 6+ default tools                                      │
│  • Authentication and rate limiting                                         │
│  • Context sharing between clients/servers                                  │
│  • Agent-to-agent communication                                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Hook Manager Page (`/hooks`)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Hook Manager                                        │
│  Manage event hooks and plugins for extensible functionality                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  🔗 Event Hooks                                            [+ Add Hook]     │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Log LLM Request              [pre] [Priority: 10]                      │ │
│  │ llm.generate.pre                        [▶️] [Enabled] [🗑️]            │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Cache Response               [post] [Priority: 5]                      │ │
│  │ llm.generate.post                       [▶️] [Enabled] [🗑️]            │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Validate Input               [pre] [Priority: 20]                      │ │
│  │ rag.query.pre                           [▶️] [Enabled] [🗑️]            │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ Track Analytics              [post] [Priority: 1]                      │ │
│  │ conversation.message.post               [▶️] [Enabled] [🗑️]            │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  📦 Plugins                                                                  │
│  ┌──────────────────────────────┐  ┌──────────────────────────────┐        │
│  │ Analytics Plugin             │  │ Caching Plugin               │        │
│  │ Track usage analytics        │  │ Cache responses              │        │
│  │ 5 hooks registered           │  │ 3 hooks registered           │        │
│  │                   [Enabled]  │  │                   [Enabled]  │        │
│  └──────────────────────────────┘  └──────────────────────────────┘        │
│  ┌──────────────────────────────┐                                           │
│  │ Logging Plugin               │                                           │
│  │ Enhanced logging             │                                           │
│  │ 8 hooks registered           │                                           │
│  │                  [Disabled]  │                                           │
│  └──────────────────────────────┘                                           │
│                                                                              │
│  ℹ️ Hook System Features:                                                   │
│  • Event-driven architecture with 40+ event types                           │
│  • Pre/post hooks for all operations                                        │
│  • Priority-based execution (0-100)                                         │
│  • Conditional hook execution                                               │
│  • Plugin system for extensibility                                          │
│  • Middleware for event transformation                                      │
│  • Hook chains with context propagation                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Color Scheme

### Primary Colors
- **Primary Blue**: `#3b82f6` (buttons, highlights)
- **Secondary**: Olive/green accents
- **Background**: Dark gradient
- **Card**: Slightly lighter dark
- **Border**: Subtle gray

### Status Colors
- **Connected/Success**: Green (`#10b981`)
- **Error**: Red (`#ef4444`)
- **Warning**: Yellow (`#f59e0b`)
- **Info**: Blue (`#3b82f6`)

### Text Colors
- **Primary**: White
- **Muted**: Gray (`#9ca3af`)
- **Accent**: Blue on hover

---

## Icons Used

### Lucide React Icons
- **Mic** 🎤 - Voice input
- **Volume2** 🔊 - Voice output
- **Server** 🖥️ - MCP servers
- **Webhook** 🔗 - Hooks
- **Tool** 🔧 - Tools
- **Package** 📦 - Plugins
- **Zap** ⚡ - Quick actions
- **Plus** ➕ - Add new
- **Trash2** 🗑️ - Delete
- **Play** ▶️ - Test/Execute
- **Settings** ⚙️ - Settings
- **User** 👤 - Profile
- **Globe** 🌍 - Language
- **Key** 🔑 - API keys
- **Filter** 🔍 - Conditions

---

## Responsive Behavior

### Desktop (>1024px)
- Three-column layout (left sidebar, main, right sidebar)
- Full navigation visible
- All features accessible

### Tablet (768px - 1024px)
- Collapsible sidebars
- Main content takes priority
- Icons-only sidebar mode

### Mobile (<768px)
- Single column layout
- Hamburger menu for navigation
- Bottom navigation bar
- Swipe gestures for sidebars

---

## Accessibility Features

✅ **Keyboard Navigation**: All interactive elements accessible via keyboard  
✅ **ARIA Labels**: Descriptive labels for screen readers  
✅ **Focus Indicators**: Clear focus states  
✅ **Color Contrast**: WCAG AA compliant  
✅ **Semantic HTML**: Proper heading hierarchy  
✅ **Alt Text**: Images and icons have descriptions  

---

## Animation & Transitions

- **Sidebar Collapse**: 300ms ease transition
- **Button Hover**: Smooth color transition
- **Audio Visualization**: Real-time 100ms updates
- **Toast Notifications**: Slide in from top-right
- **Page Transitions**: Fade in/out

---

## Next Steps

1. ✅ Test all pages in browser
2. ✅ Verify responsive behavior
3. ✅ Check dark mode consistency
4. ✅ Test keyboard navigation
5. ✅ Connect to backend APIs
6. ✅ Add loading states
7. ✅ Add error handling
8. ✅ User testing and feedback

**The UI is ready for testing!** 🎉

