# CPAS4 UI Implementation Guide

## Overview

This document provides a comprehensive guide to the CPAS4 UI implementation, including architecture, components, and integration with the CPAS4 backend.

## Architecture

### Technology Stack

- **Frontend Framework**: React 18.3 with TypeScript 5.6
- **Build Tool**: Vite 7.1.9
- **Styling**: Tailwind CSS 4.0 with @tailwindcss/postcss
- **UI Components**: shadcn/ui (Radix UI primitives)
- **State Management**: Zustand 4.x with persist middleware
- **Icons**: Lucide React
- **HTTP Client**: Axios
- **Type Safety**: Full TypeScript coverage

### Project Structure

```
cpas4-ui/
├── src/
│   ├── components/
│   │   ├── ui/                      # shadcn/ui base components
│   │   │   ├── button.tsx           # Button component
│   │   │   ├── input.tsx            # Input component
│   │   │   ├── dialog.tsx           # Dialog/Modal component
│   │   │   ├── tooltip.tsx          # Tooltip component
│   │   │   └── slider.tsx           # Slider component
│   │   ├── layout/                  # Layout components
│   │   │   ├── Header.tsx           # Top header with theme toggle
│   │   │   ├── LeftSidebar.tsx      # Agent types & quick actions
│   │   │   └── RightSidebar.tsx     # Task history & system status
│   │   ├── agents/                  # Agent-related components
│   │   │   └── MasterAgentContext.tsx  # Main chat interface
│   │   └── shared/                  # Shared components
│   │       └── SettingsDialog.tsx   # Settings modal
│   ├── lib/
│   │   ├── api.ts                   # CPAS4 API client
│   │   └── utils.ts                 # Utility functions (cn)
│   ├── stores/
│   │   └── uiStore.ts               # Zustand store for UI state
│   ├── types/
│   │   └── index.ts                 # TypeScript type definitions
│   ├── App.tsx                      # Main application component
│   ├── main.tsx                     # Application entry point
│   └── index.css                    # Global styles & Tailwind
├── public/
│   └── backgrounds/                 # User background images
├── .env                             # Environment variables
├── tailwind.config.js               # Tailwind configuration
├── postcss.config.js                # PostCSS configuration
├── tsconfig.json                    # TypeScript configuration
├── vite.config.ts                   # Vite configuration
└── package.json                     # Dependencies
```

## Color Scheme

### Light Mode
- **Primary (Blue)**: `hsl(217.2, 91.2%, 59.8%)` - #3B82F6
- **Secondary (Olive)**: `hsl(82.7, 84.5%, 67.1%)` - #A3E635
- **Background**: `hsl(0, 0%, 100%)` - White
- **Foreground**: `hsl(222.2, 84%, 4.9%)` - Dark gray

### Dark Mode
- **Primary (Blue)**: `hsl(217.2, 91.2%, 59.8%)` - #3B82F6
- **Secondary (Olive)**: `hsl(82.7, 84.5%, 42.8%)` - #65A30D
- **Background**: `hsl(222.2, 84%, 4.9%)` - Very dark blue
- **Foreground**: `hsl(210, 40%, 98%)` - Off-white

## Key Components

### 1. Header Component

**Location**: `src/components/layout/Header.tsx`

**Features**:
- CPAS4 branding with gradient text
- Theme toggle (light/dark mode)
- Settings button
- Responsive design

**State Management**:
- Uses `useUIStore` for theme state
- Persists theme preference in localStorage

### 2. Master Agent Context Window

**Location**: `src/components/agents/MasterAgentContext.tsx`

**Features**:
- Large, prominent chat interface
- Custom background image support
- Drag-and-drop file upload
- Prompt enhancement button
- File upload button
- Auto-scroll to latest message
- Loading indicators

**Background System**:
- Opaque background image with adjustable opacity overlay
- Default blue gradient when no image selected
- Background stored in localStorage via Zustand

**Action Icons**:
- **Sparkles Icon**: Enhance prompt with AI (connects to `/api/v1/prompts/enhance`)
- **Upload Icon**: Upload documents, files, images, or links (connects to `/api/v1/knowledge/*`)

### 3. Left Sidebar

**Location**: `src/components/layout/LeftSidebar.tsx`

**Features**:
- 11 agent types with emoji icons
- Quick actions menu
- Recent activity section
- Collapsible with state persistence
- Smooth transitions

**Agent Types**:
1. Research 🔍
2. Coding 💻
3. Data Analysis 📊
4. Content Creation ✍️
5. Task Automation ⚙️
6. Customer Support 💬
7. Sales 💰
8. Marketing 📢
9. HR 👥
10. Finance 💵
11. General 🤖

### 4. Right Sidebar

**Location**: `src/components/layout/RightSidebar.tsx`

**Features**:
- Task history
- Active jobs monitoring
- System status indicators
- Collapsible with state persistence

**System Status**:
- Backend connection status
- Database connection status
- Ollama LLM status

### 5. Settings Dialog

**Location**: `src/components/shared/SettingsDialog.tsx`

**Features**:
- Background image upload
- Background opacity slider
- Image preview
- API configuration display
- Remove background option

**File Support**:
- Formats: JPG, PNG, GIF
- Recommended size: 1920x1080px
- Stored as base64 in localStorage

## State Management

### UI Store (Zustand)

**Location**: `src/stores/uiStore.ts`

**State**:
```typescript
interface UIStore {
  theme: 'light' | 'dark';
  backgroundImage?: string;
  backgroundOpacity: number;
  leftSidebarCollapsed: boolean;
  rightSidebarCollapsed: boolean;
}
```

**Actions**:
- `setTheme(theme)`: Set theme and apply to DOM
- `toggleTheme()`: Toggle between light/dark
- `setBackgroundImage(image)`: Set background image (base64)
- `setBackgroundOpacity(opacity)`: Set opacity (0-1)
- `toggleLeftSidebar()`: Toggle left sidebar
- `toggleRightSidebar()`: Toggle right sidebar

**Persistence**:
- All state persisted to localStorage
- Key: `cpas4-ui-preferences`
- Automatically rehydrated on app load

## API Integration

### API Client

**Location**: `src/lib/api.ts`

**Base URL**: `http://localhost:8000/api/v1` (configurable via `.env`)

**Endpoints**:

```typescript
// Agent Management
agentAPI.list()                    // GET /agents
agentAPI.get(id)                   // GET /agents/:id
agentAPI.create(data)              // POST /agents
agentAPI.delete(id)                // DELETE /agents/:id
agentAPI.update(id, data)          // PUT /agents/:id

// Task Execution
taskAPI.execute(data)              // POST /tasks/execute
taskAPI.get(id)                    // GET /tasks/:id
taskAPI.list()                     // GET /tasks
taskAPI.cancel(id)                 // POST /tasks/:id/cancel

// Prompt Enhancement
promptAPI.enhance(data)            // POST /prompts/enhance
promptAPI.enhanceAsync(data)       // POST /prompts/enhance/async
promptAPI.get(id)                  // GET /prompts/enhance/:id
promptAPI.feedback(id, data)       // POST /prompts/enhance/:id/feedback
promptAPI.statistics()             // GET /prompts/statistics

// Knowledge Base
knowledgeAPI.upload(formData)      // POST /knowledge/upload
knowledgeAPI.search(query, params) // POST /knowledge/search
knowledgeAPI.list()                // GET /knowledge/documents
knowledgeAPI.delete(id)            // DELETE /knowledge/documents/:id

// Reports
reportAPI.generate(data)           // POST /reports/generate
reportAPI.get(id)                  // GET /reports/:id
reportAPI.list()                   // GET /reports
reportAPI.download(id)             // GET /reports/:id/download
reportAPI.statistics()             // GET /reports/statistics

// Scraping
scrapingAPI.createJob(data)        // POST /scraping/jobs
scrapingAPI.getJob(id)             // GET /scraping/jobs/:id
scrapingAPI.listJobs()             // GET /scraping/jobs
scrapingAPI.cancelJob(id)          // POST /scraping/jobs/:id/cancel
```

## Type Definitions

**Location**: `src/types/index.ts`

**Key Types**:
- `Agent`: Agent entity
- `AgentType`: 11 agent type literals
- `Task`: Task entity
- `TaskStatus`: Task status literals
- `Message`: Chat message
- `PromptEnhancement`: Enhanced prompt data
- `Report`: Report entity
- `Document`: Knowledge base document
- `ScrapingJob`: Scraping job entity
- `UIPreferences`: UI state preferences

## Responsive Design

### Breakpoints

- **Mobile**: < 640px (sidebars collapse)
- **Tablet**: 640px - 1024px (sidebars collapsible)
- **Desktop**: > 1024px (full three-column layout)

### Layout Behavior

- **Mobile**: Single column, sidebars as overlays
- **Tablet**: Two columns, one sidebar visible
- **Desktop**: Three columns, both sidebars visible

## Development Workflow

### Running Locally

```bash
# Install dependencies
npm install

# Start dev server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### Environment Variables

Create `.env` file:

```env
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

### Adding New Components

1. Create component in appropriate directory
2. Use TypeScript for type safety
3. Follow shadcn/ui patterns
4. Use Tailwind CSS for styling
5. Export from component file

### Code Style

- **Components**: PascalCase (e.g., `MasterAgentContext.tsx`)
- **Utilities**: camelCase (e.g., `api.ts`)
- **Types**: PascalCase interfaces (e.g., `Agent`)
- **CSS**: Tailwind utility classes
- **State**: Zustand stores with TypeScript

## Next Steps

### Phase 2: Core Features (Weeks 2-4)

1. **Prompt Enhancement Integration**
   - Connect sparkles button to `/api/v1/prompts/enhance`
   - Display quality metrics
   - User feedback interface

2. **File Upload Integration**
   - Connect upload button to `/api/v1/knowledge/upload`
   - Progress indicators
   - Success/error notifications

3. **Task Execution**
   - Real-time task monitoring
   - Task history display
   - Cancel task functionality

4. **Agent Management**
   - Create new agents
   - Edit agent configuration
   - Delete agents

### Phase 3: Advanced Features (Weeks 5-8)

5. **Report Visualization**
   - Display generated reports
   - Interactive Tufte visualizations
   - Download reports

6. **Multi-Agent Orchestration**
   - Visual workflow builder
   - Agent coordination UI

7. **WebSocket Integration**
   - Real-time task updates
   - Live report generation progress

### Phase 4: Desktop App (Weeks 9-13)

8. **Tauri Integration**
   - Package as desktop app
   - Native notifications
   - System tray integration

## Troubleshooting

### Common Issues

**Issue**: Tailwind styles not applying
**Solution**: Ensure `@tailwindcss/postcss` is installed and configured in `postcss.config.js`

**Issue**: Theme not persisting
**Solution**: Check browser localStorage for `cpas4-ui-preferences` key

**Issue**: API calls failing
**Solution**: Verify CPAS4 backend is running on `http://localhost:8000`

**Issue**: Background image not displaying
**Solution**: Check image format (JPG, PNG, GIF) and size (< 5MB recommended)

## Performance Optimization

### Best Practices

1. **Code Splitting**: Use React.lazy() for route-based splitting
2. **Memoization**: Use React.memo() for expensive components
3. **Virtual Scrolling**: Implement for long message lists
4. **Image Optimization**: Compress background images
5. **Bundle Size**: Monitor with `npm run build`

## Security Considerations

1. **API Keys**: Never commit `.env` files
2. **XSS Protection**: Sanitize user input
3. **CORS**: Configure backend CORS properly
4. **File Upload**: Validate file types and sizes
5. **localStorage**: Don't store sensitive data

## License

Part of the CPAS4 project.

