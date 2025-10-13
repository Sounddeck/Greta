# CPAS4 UI - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Prerequisites

- ✅ Node.js 18+ installed
- ✅ CPAS4 backend running on `http://localhost:8000`

### Installation

```bash
# Navigate to UI directory
cd cpas4-ui

# Install dependencies
npm install

# Start development server
npm run dev
```

Open your browser to **http://localhost:5173**

## 🎨 First Steps

### 1. Explore the Interface

The UI has three main areas:

- **Left Sidebar**: Agent types and quick actions
- **Center**: Master Agent chat interface
- **Right Sidebar**: Task history and system status

### 2. Toggle Dark Mode

Click the **moon/sun icon** in the top-right header to switch themes.

### 3. Customize Background

1. Click the **gear icon** (Settings) in the header
2. Click **"Upload Background"**
3. Select an image from your computer
4. Adjust opacity with the slider
5. Close settings to see your custom background

### 4. Start a Conversation

1. Type a message in the input box at the bottom
2. Press **Enter** to send
3. Watch the AI respond!

### 5. Enhance Your Prompts

1. Type a prompt in the input box
2. Click the **sparkles icon** ✨ to enhance it with AI
3. See your prompt improved automatically

### 6. Upload Files

**Method 1**: Click the upload icon 📤
**Method 2**: Drag and drop files into the chat area

Supported formats: PDF, TXT, DOCX, PNG, JPG, JPEG, GIF

## 🎯 Key Features

### Collapsible Sidebars

- Click the **chevron buttons** on sidebars to collapse/expand
- Your preference is saved automatically

### Agent Types

Choose from 11 specialized agents:
- 🔍 Research
- 💻 Coding
- 📊 Data Analysis
- ✍️ Content Creation
- ⚙️ Task Automation
- 💬 Customer Support
- 💰 Sales
- 📢 Marketing
- 👥 HR
- 💵 Finance
- 🤖 General

### Keyboard Shortcuts

- **Enter**: Send message
- **Shift + Enter**: New line in message

## 🔧 Configuration

### Change Backend URL

Edit `.env` file:

```env
VITE_API_BASE_URL=http://your-backend-url/api/v1
```

Restart the dev server after changing.

## 📱 Responsive Design

The UI works on:
- 💻 Desktop (full three-column layout)
- 📱 Tablet (two-column layout)
- 📱 Mobile (single column with overlay sidebars)

## 🐛 Troubleshooting

### UI not loading?

1. Check that dev server is running: `npm run dev`
2. Verify you're on `http://localhost:5173`
3. Check browser console for errors (F12)

### Backend connection failed?

1. Ensure CPAS4 backend is running on port 8000
2. Check `.env` file has correct `VITE_API_BASE_URL`
3. Verify CORS is configured on backend

### Styles not working?

1. Clear browser cache (Ctrl+Shift+R or Cmd+Shift+R)
2. Delete `node_modules` and run `npm install` again
3. Check that `@tailwindcss/postcss` is installed

### Background image not showing?

1. Check image format (JPG, PNG, GIF only)
2. Try a smaller image (< 5MB)
3. Check browser console for errors

## 🎓 Next Steps

### Learn More

- Read `IMPLEMENTATION_GUIDE.md` for detailed architecture
- Check `README.md` for full feature list
- Explore `src/components/` to understand component structure

### Customize

- Edit `src/index.css` to change colors
- Modify `tailwind.config.js` for theme customization
- Add new components in `src/components/`

### Integrate with Backend

- Review `src/lib/api.ts` for API endpoints
- Check `src/types/index.ts` for type definitions
- Implement real API calls in components

## 💡 Tips

1. **Use Dark Mode**: Easier on the eyes for long sessions
2. **Collapse Sidebars**: Get more space for the chat interface
3. **Upload Documents**: Build your knowledge base for better AI responses
4. **Enhance Prompts**: Get better results with AI-improved prompts
5. **Save Backgrounds**: Your custom background persists across sessions

## 🆘 Need Help?

- Check the browser console (F12) for errors
- Review the `IMPLEMENTATION_GUIDE.md` for detailed docs
- Ensure CPAS4 backend is running and accessible

## 🎉 You're Ready!

Start chatting with your AI agents and explore the power of CPAS4!

---

**Happy Automating! 🤖**

