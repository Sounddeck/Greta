# CPAS4 UI Layout Guide

## Visual Layout Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              HEADER (64px)                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  CPAS4 Logo          Comprehensive Personal Automation System         │  │
│  │                                                    🌙 Theme  ⚙️ Settings │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┬────────────────────────────────────────────┬──────────┐     │
│  │          │                                             │          │     │
│  │   LEFT   │           CENTER AREA                       │  RIGHT   │     │
│  │ SIDEBAR  │      (Master Agent Context)                 │ SIDEBAR  │     │
│  │          │                                             │          │     │
│  │  280px   │            Flexible Width                   │  280px   │     │
│  │          │                                             │          │     │
│  │ ┌──────┐ │  ┌────────────────────────────────────┐   │ ┌──────┐ │     │
│  │ │Agent │ │  │                                     │   │ │Task  │ │     │
│  │ │Types │ │  │   Background Image (Opaque)         │   │ │Hist. │ │     │
│  │ │      │ │  │   + Opacity Overlay                 │   │ │      │ │     │
│  │ │🔍    │ │  │                                     │   │ │      │ │     │
│  │ │💻    │ │  │   ┌─────────────────────────────┐  │   │ ├──────┤ │     │
│  │ │📊    │ │  │   │  Master Agent               │  │   │ │Active│ │     │
│  │ │✍️    │ │  │   │  Chat Interface             │  │   │ │Jobs  │ │     │
│  │ │⚙️    │ │  │   │                             │  │   │ │      │ │     │
│  │ │💬    │ │  │   │  [Message History]          │  │   │ │      │ │     │
│  │ │💰    │ │  │   │                             │  │   │ ├──────┤ │     │
│  │ │📢    │ │  │   │  User: Hello!               │  │   │ │Sys   │ │     │
│  │ │👥    │ │  │   │  AI: Hi there!              │  │   │ │Status│ │     │
│  │ │💵    │ │  │   │                             │  │   │ │      │ │     │
│  │ │🤖    │ │  │   │                             │  │   │ │✅ API│ │     │
│  │ └──────┘ │  │   │                             │  │   │ │✅ DB │ │     │
│  │          │  │   │                             │  │   │ │✅ LLM│ │     │
│  │ ┌──────┐ │  │   └─────────────────────────────┘  │   │ └──────┘ │     │
│  │ │Quick │ │  │                                     │   │          │     │
│  │ │Action│ │  │   ┌─────────────────────────────┐  │   │          │     │
│  │ │      │ │  │   │ ✨ 📤  [Type message...]  ➤│  │   │          │     │
│  │ │New   │ │  │   └─────────────────────────────┘  │   │          │     │
│  │ │Agent │ │  │                                     │   │          │     │
│  │ │      │ │  └────────────────────────────────────┘   │          │     │
│  │ │Exec  │ │                                           │          │     │
│  │ │Task  │ │                                           │          │     │
│  │ │      │ │                                           │          │     │
│  │ │Report│ │                                           │          │     │
│  │ └──────┘ │                                           │          │     │
│  │          │                                           │          │     │
│  │ ┌──────┐ │                                           │          │     │
│  │ │Recent│ │                                           │          │     │
│  │ │      │ │                                           │          │     │
│  │ │(None)│ │                                           │          │     │
│  │ └──────┘ │                                           │          │     │
│  │          │                                           │          │     │
│  │  ◀ ▶    │                                           │  ◀ ▶      │     │
│  │ Collapse │                                           │ Collapse  │     │
│  └──────────┴────────────────────────────────────────────┴──────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### Header (Fixed, 64px height)

```
┌─────────────────────────────────────────────────────────────────┐
│  CPAS4                    Comprehensive Personal Automation     │
│  [Logo]                                          🌙  ⚙️         │
└─────────────────────────────────────────────────────────────────┘
```

**Elements**:
- Left: CPAS4 logo with gradient (blue → olive)
- Center: Subtitle text
- Right: Theme toggle (moon/sun) + Settings button

### Left Sidebar (280px, collapsible)

```
┌──────────────┐
│ 🤖 Agent Types│
├──────────────┤
│ 🔍 Research  │
│ 💻 Coding    │
│ 📊 Data Anal.│
│ ✍️ Content   │
│ ⚙️ Task Auto │
│ 💬 Support   │
│ 💰 Sales     │
│ 📢 Marketing │
│ 👥 HR        │
│ 💵 Finance   │
│ 🤖 General   │
├──────────────┤
│ ⚡ Quick Acts │
├──────────────┤
│ New Agent    │
│ Execute Task │
│ Gen. Report  │
├──────────────┤
│ 🕐 Recent    │
├──────────────┤
│ (No activity)│
└──────────────┘
   ◀ Collapse
```

**Sections**:
1. Agent Types (11 items with emoji icons)
2. Quick Actions (3 buttons)
3. Recent Activity (placeholder)
4. Collapse button (bottom)

### Center Area (Flexible width)

```
┌────────────────────────────────────────────────┐
│                                                 │
│  [Background Image - Opaque]                   │
│  [Opacity Overlay - Adjustable]                │
│                                                 │
│  ┌──────────────────────────────────────────┐ │
│  │ Master Agent                    0 messages│ │
│  ├──────────────────────────────────────────┤ │
│  │                                           │ │
│  │  [Empty State]                            │ │
│  │                                           │ │
│  │         🤖                                │ │
│  │  Welcome to CPAS4                         │ │
│  │  Start a conversation...                  │ │
│  │                                           │ │
│  │  OR                                       │ │
│  │                                           │ │
│  │  [Message History]                        │ │
│  │                                           │ │
│  │  ┌─────────────────────────────────────┐ │ │
│  │  │ User: Hello!              10:30 AM  │ │ │
│  │  └─────────────────────────────────────┘ │ │
│  │                                           │ │
│  │  ┌─────────────────────────────────────┐ │ │
│  │  │ AI: Hi! How can I help?   10:30 AM  │ │ │
│  │  └─────────────────────────────────────┘ │ │
│  │                                           │ │
│  ├──────────────────────────────────────────┤ │
│  │ ✨ 📤  [Type your message...]        ➤  │ │
│  └──────────────────────────────────────────┘ │
│                                                 │
└────────────────────────────────────────────────┘
```

**Elements**:
- Background layer (opaque image or gradient)
- Overlay layer (adjustable opacity)
- Chat window (rounded, semi-transparent)
- Header (title + message count)
- Message area (scrollable)
- Input area (icons + text + send button)

**Action Icons**:
- ✨ Sparkles: Enhance prompt
- 📤 Upload: Upload files

### Right Sidebar (280px, collapsible)

```
┌──────────────┐
│ 🕐 Task Hist.│
├──────────────┤
│ (No tasks)   │
│              │
├──────────────┤
│ ⚡ Active Jobs│
├──────────────┤
│ (No jobs)    │
│              │
├──────────────┤
│ ✅ System    │
├──────────────┤
│ Backend  ✅  │
│ Database ✅  │
│ Ollama   ✅  │
└──────────────┘
   Collapse ▶
```

**Sections**:
1. Task History (placeholder)
2. Active Jobs (placeholder)
3. System Status (3 indicators)
4. Collapse button (bottom)

## Color Scheme

### Light Mode

```
┌─────────────────────────────────────────┐
│ Background:  White (#FFFFFF)            │
│ Foreground:  Dark Gray (#0F172A)        │
│ Primary:     Blue (#3B82F6)             │
│ Secondary:   Olive (#A3E635)            │
│ Border:      Light Gray (#E2E8F0)       │
│ Muted:       Gray (#64748B)             │
└─────────────────────────────────────────┘
```

### Dark Mode

```
┌─────────────────────────────────────────┐
│ Background:  Very Dark Blue (#0F172A)   │
│ Foreground:  Off-White (#F8FAFC)        │
│ Primary:     Blue (#3B82F6)             │
│ Secondary:   Olive (#65A30D)            │
│ Border:      Dark Gray (#1E293B)        │
│ Muted:       Light Gray (#94A3B8)       │
└─────────────────────────────────────────┘
```

## Responsive Breakpoints

### Desktop (> 1024px)

```
┌─────────────────────────────────────────────────┐
│              HEADER                              │
├──────────┬─────────────────────┬─────────────────┤
│   LEFT   │      CENTER         │     RIGHT       │
│  SIDEBAR │   (Master Agent)    │    SIDEBAR      │
│  280px   │    Flexible         │     280px       │
└──────────┴─────────────────────┴─────────────────┘
```

### Tablet (640px - 1024px)

```
┌─────────────────────────────────────────┐
│              HEADER                      │
├──────────┬──────────────────────────────┤
│   LEFT   │         CENTER               │
│  SIDEBAR │      (Master Agent)          │
│  240px   │       Flexible               │
│          │                              │
│  (Right  │                              │
│  sidebar │                              │
│  hidden) │                              │
└──────────┴──────────────────────────────┘
```

### Mobile (< 640px)

```
┌─────────────────────────┐
│        HEADER           │
├─────────────────────────┤
│                         │
│       CENTER            │
│   (Master Agent)        │
│     Full Width          │
│                         │
│  (Sidebars as overlays) │
│                         │
└─────────────────────────┘
```

## Settings Dialog

```
┌────────────────────────────────────────┐
│  Settings                          ✕   │
├────────────────────────────────────────┤
│                                        │
│  🖼️ Background Image                  │
│  ┌──────────────────────────────────┐ │
│  │                                   │ │
│  │   [Image Preview]                 │ │
│  │                                   │ │
│  └──────────────────────────────────┘ │
│                                        │
│  [Upload Background]  [Remove]         │
│                                        │
│  Background Opacity            30%     │
│  ├────●─────────────────────────────┤ │
│                                        │
│  API Configuration                     │
│  Backend URL: localhost:8000           │
│                                        │
└────────────────────────────────────────┘
```

## Interaction States

### Sidebar Collapsed (Left)

```
┌──┐
│🤖│
│🔍│
│💻│
│📊│
│✍️│
│⚙️│
│💬│
│💰│
│📢│
│👥│
│💵│
│🤖│
│  │
│▶ │
└──┘
16px
```

### Sidebar Expanded (Left)

```
┌──────────────┐
│ 🤖 Agent Types│
│ 🔍 Research  │
│ 💻 Coding    │
│ 📊 Data Anal.│
│ ✍️ Content   │
│ ⚙️ Task Auto │
│ 💬 Support   │
│ 💰 Sales     │
│ 📢 Marketing │
│ 👥 HR        │
│ 💵 Finance   │
│ 🤖 General   │
│              │
│ ◀            │
└──────────────┘
    280px
```

## Drag-and-Drop States

### Normal State

```
┌────────────────────────────────────┐
│                                     │
│  [Chat messages]                    │
│                                     │
└────────────────────────────────────┘
```

### Dragging Over

```
┌────────────────────────────────────┐
│ ╔═══════════════════════════════╗ │
│ ║                                ║ │
│ ║  Drop files here to upload     ║ │
│ ║                                ║ │
│ ╚═══════════════════════════════╝ │
└────────────────────────────────────┘
  (Blue dashed border, light blue bg)
```

## Message Bubbles

### User Message (Right-aligned)

```
                    ┌─────────────────────┐
                    │ Hello, how are you? │
                    │           10:30 AM  │
                    └─────────────────────┘
                         (Blue background)
```

### AI Message (Left-aligned)

```
┌─────────────────────┐
│ I'm doing great!    │
│ How can I help?     │
│           10:30 AM  │
└─────────────────────┘
   (Gray background)
```

## Loading States

### Message Loading

```
┌─────────────────────┐
│  ⚪ ⚪ ⚪            │
│  (Animated dots)    │
└─────────────────────┘
```

### Button Loading

```
┌──────────────┐
│  ⚪ Loading  │
└──────────────┘
  (Spinning icon)
```

---

This layout guide provides a visual reference for the CPAS4 UI structure and responsive behavior.

