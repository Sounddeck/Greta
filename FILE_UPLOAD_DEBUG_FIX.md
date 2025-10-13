# 🔧 File Upload Debugging & Fix

**Date:** October 13, 2025  
**Issue:** File upload failed when testing from standalone HTML page  
**Status:** ✅ RESOLVED  

---

## Problem Identified

### Root Cause: CORS Policy Violation

The file upload test page (`test_file_upload.html`) was initially opened using the `file://` protocol, which caused a CORS (Cross-Origin Resource Sharing) error when trying to access the backend API at `http://localhost:8000`.

**Error Details:**
- **Protocol Mismatch:** `file://` → `http://localhost:8000`
- **CORS Configuration:** Backend only allows specific origins:
  - `http://localhost:5173`
  - `http://localhost:5174`
  - `http://localhost:3000`
- **Result:** Browser blocked the request due to CORS policy

---

## Backend CORS Configuration

**File:** `backend/main.py` (Lines 348-360)

```python
# Add CORS middleware - Environment-based configuration
allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:5174,http://localhost:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Environment-based instead of ["*"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Why This Is Correct:**
- ✅ Secure: Only allows specific trusted origins
- ✅ Environment-based: Can be configured via `.env` file
- ✅ Includes all development ports (5173, 5174, 3000)
- ❌ Does NOT allow `file://` protocol (by design, for security)

---

## Solution Applied

### Fix: Serve Test Page from Vite Dev Server

**Action Taken:**
```bash
mv Desktop/CPAS5/test_file_upload.html Desktop/CPAS5/cpas4-ui/public/test_file_upload.html
```

**Result:**
- Test page now accessible at: `http://localhost:5174/test_file_upload.html`
- Same origin as Vite dev server: `http://localhost:5174`
- CORS policy satisfied ✅

---

## Verification Steps

### 1. Backend Endpoint Test ✅

**Command:**
```bash
curl -X POST http://localhost:8000/api/v1/knowledge/upload \
  -F "files=@/tmp/test_upload_debug.txt" \
  -F "collection=test_debug"
```

**Response:**
```json
{
  "success": true,
  "files": [{
    "id": "doc_1760362347.375592",
    "filename": "test_upload_debug.txt",
    "file_type": "text/plain",
    "file_size": 29,
    "collection": "test_debug",
    "uploaded_at": "2025-10-13T09:32:27.375773",
    "metadata": {
      "original_filename": "test_upload_debug.txt",
      "saved_path": "data/uploads/1760362347.375592_test_upload_debug.txt",
      "content_type": "text/plain"
    }
  }],
  "total": 1,
  "message": "Successfully uploaded 1 file(s)"
}
```

**Status:** ✅ Backend working perfectly

### 2. Frontend Code Verification ✅

**File:** `cpas4-ui/src/components/agents/MasterAgentContext.tsx`

**Changes Made:**
- ✅ Added `knowledgeAPI` import
- ✅ Added `uploadedFiles` state
- ✅ Implemented `handleFileUpload` function
- ✅ Implemented `handleDrop` function for drag-and-drop
- ✅ Added uploaded files display section
- ✅ Added success/error messages to chat

**TypeScript Diagnostics:** ✅ No errors

### 3. API Client Verification ✅

**File:** `cpas4-ui/src/lib/api.ts` (Lines 48-54)

```typescript
export const knowledgeAPI = {
  upload: (formData: FormData) => api.post('/knowledge/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  }),
  search: (query: string, params?: any) => api.post('/knowledge/search', { query, ...params }),
  // ... other methods
};
```

**Status:** ✅ Correct implementation

---

## How to Test File Upload

### Method 1: Standalone Test Page

**URL:** http://localhost:5174/test_file_upload.html

**Steps:**
1. Open the URL in your browser
2. Click the upload area or drag files
3. Select one or more files
4. Click "Upload Files" button
5. Verify success message appears
6. Check uploaded files list

**Expected Result:**
- ✅ Files upload successfully
- ✅ Success message displayed
- ✅ File list shows uploaded files with names and sizes
- ✅ Files saved to `backend/data/uploads/` directory

### Method 2: Greta UI (Recommended)

**URL:** http://localhost:5174/

**Steps:**
1. Open Greta application
2. Navigate to any agent (Research, Coding, Data Analysis, etc.)
3. Click the **Upload** button (📎 icon) in chat input area
4. Select files to upload
5. Verify success message in chat
6. Check uploaded files section above chat

**Expected Result:**
- ✅ Upload button visible and clickable
- ✅ File picker opens
- ✅ Files upload automatically after selection
- ✅ Success message appears in chat: "✅ Successfully uploaded X file(s): filename1, filename2..."
- ✅ Uploaded files section displays with file names and sizes
- ✅ Remove button (×) works for each file

### Method 3: Drag and Drop

**Steps:**
1. Open Greta UI at http://localhost:5174/
2. Navigate to any agent
3. Drag files from your file system
4. Drop them into the messages area
5. Verify upload and feedback

**Expected Result:**
- ✅ Drag-over visual feedback
- ✅ Files upload on drop
- ✅ Success message in chat: "✅ Successfully uploaded X file(s) via drag & drop: ..."
- ✅ Files appear in uploaded files section

---

## Common Issues & Solutions

### Issue 1: CORS Error
**Symptom:** "Access to fetch at 'http://localhost:8000/...' from origin '...' has been blocked by CORS policy"

**Solution:**
- Ensure you're accessing the app from `http://localhost:5174/` (not `file://`)
- Check backend is running on port 8000
- Verify CORS configuration in `backend/main.py`

### Issue 2: Backend Not Running
**Symptom:** "Failed to fetch" or "Network error"

**Solution:**
```bash
# Check if backend is running
ps aux | grep uvicorn

# If not running, start it
cd Desktop/CPAS5/backend
source venv/bin/activate  # or your venv path
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Issue 3: Frontend Not Running
**Symptom:** Cannot access http://localhost:5174/

**Solution:**
```bash
# Check if frontend is running
ps aux | grep vite

# If not running, start it
cd Desktop/CPAS5/cpas4-ui
npm run dev
```

### Issue 4: File Not Uploading
**Symptom:** No error, but file doesn't upload

**Solution:**
- Check file type is supported: `.pdf, .txt, .docx, .png, .jpg, .jpeg, .gif`
- Check file size (no explicit limit, but large files may timeout)
- Check browser console for JavaScript errors
- Verify `data/uploads/` directory exists and is writable

---

## File Upload Flow

### Frontend → Backend Flow

```
1. User selects file(s)
   ↓
2. handleFileUpload() or handleDrop() triggered
   ↓
3. Create FormData with files
   ↓
4. Call knowledgeAPI.upload(formData)
   ↓
5. POST /api/v1/knowledge/upload
   ↓
6. Backend receives files
   ↓
7. Save to data/uploads/ with timestamp prefix
   ↓
8. Return JSON response with file metadata
   ↓
9. Frontend updates uploadedFiles state
   ↓
10. Display success message in chat
   ↓
11. Show files in uploaded files section
```

---

## Files Modified/Created

### Modified
- `cpas4-ui/src/components/agents/MasterAgentContext.tsx` - Added file upload functionality

### Created
- `cpas4-ui/public/test_file_upload.html` - Standalone test page (moved from root)
- `FILE_UPLOAD_DEBUG_FIX.md` - This document

### Unchanged (Already Working)
- `backend/routers/knowledge_router.py` - Upload endpoint
- `cpas4-ui/src/lib/api.ts` - API client
- `backend/main.py` - CORS configuration

---

## Testing Checklist

- [x] Backend endpoint tested with curl ✅
- [x] CORS configuration verified ✅
- [x] Frontend code has no TypeScript errors ✅
- [x] Test page moved to Vite server ✅
- [x] Test page accessible at http://localhost:5174/test_file_upload.html ✅
- [ ] **User to test:** Upload via test page
- [ ] **User to test:** Upload via Greta UI button
- [ ] **User to test:** Upload via drag-and-drop
- [ ] **User to test:** Verify files saved to backend/data/uploads/
- [ ] **User to test:** Verify uploaded files display correctly
- [ ] **User to test:** Verify remove button works

---

## Conclusion

**Issue:** CORS policy blocked file upload from `file://` protocol  
**Fix:** Moved test page to Vite dev server (`http://localhost:5174/`)  
**Status:** ✅ RESOLVED  

**Next Steps:**
1. User tests file upload functionality
2. If working, proceed to Week 1, Item 3 (Prompt Library Integration)
3. Continue with remaining roadmap items

---

**Test URLs:**
- **Test Page:** http://localhost:5174/test_file_upload.html
- **Greta UI:** http://localhost:5174/
- **Backend API:** http://localhost:8000/docs (Swagger UI)

**Servers Running:**
- ✅ Backend: Terminal 1533 (port 8000)
- ✅ Frontend: Terminal 1019 (port 5174)

