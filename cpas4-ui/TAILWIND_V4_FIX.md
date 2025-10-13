# Tailwind CSS 4.0 Configuration Fix

## Problem

When running the CPAS4 UI development server, the following error occurred:

```
Error: Cannot apply unknown utility class `border-border`. 
Are you using CSS modules or similar and missing `@reference`?
```

**Error Location**: `src/index.css` at line 1 (the `@tailwind base;` directive)

## Root Cause

The project was initially configured with **Tailwind CSS 4.0** syntax mixed with **Tailwind CSS 3.x** patterns:

1. **PostCSS Plugin**: Using `@tailwindcss/postcss` (v4 plugin) ✅
2. **CSS File**: Using old v3 syntax with `@tailwind` directives and `@layer base` ❌
3. **Config File**: Using v3 HSL color syntax with `hsl(var(--border))` ❌

Tailwind CSS 4.0 has **breaking changes** in how CSS is structured:
- No more `@tailwind base/components/utilities` directives
- No more `@layer base` for custom properties
- New `@import "tailwindcss"` syntax
- New `@theme` directive for theme customization
- Direct CSS custom properties instead of HSL values

## Solution

### 1. Updated `src/index.css`

**Before** (Tailwind v3 syntax):
```css
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  :root {
    --background: 0 0% 100%;
    --foreground: 222.2 84% 4.9%;
    /* ... more HSL values */
  }
  
  .dark {
    --background: 222.2 84% 4.9%;
    /* ... more HSL values */
  }
}

@layer base {
  * {
    @apply border-border;  /* ❌ This caused the error */
  }
  body {
    @apply bg-background text-foreground;
  }
}
```

**After** (Tailwind v4 syntax):
```css
@import "tailwindcss";

/* Define theme colors using Tailwind v4 syntax */
@theme {
  /* Light mode colors */
  --color-background: #ffffff;
  --color-foreground: #0f172a;
  --color-primary: #3b82f6; /* Blue-500 */
  --color-secondary: #a3e635; /* Lime-400 (Olive) */
  /* ... more direct color values */
  
  /* Border radius */
  --radius-lg: 0.5rem;
  --radius-md: calc(0.5rem - 2px);
  --radius-sm: calc(0.5rem - 4px);
}

/* Dark mode colors */
.dark {
  --color-background: #0f172a;
  --color-foreground: #f8fafc;
  --color-primary: #3b82f6; /* Blue-500 */
  --color-secondary: #65a30d; /* Lime-600 (Olive) */
  /* ... more direct color values */
}

/* Base styles */
* {
  border-color: var(--color-border);  /* ✅ Direct CSS */
}

body {
  background-color: var(--color-background);
  color: var(--color-foreground);
  font-family: Inter, system-ui, Avenir, Helvetica, Arial, sans-serif;
  margin: 0;
  min-height: 100vh;
}
```

**Key Changes**:
- ✅ Replaced `@tailwind` directives with `@import "tailwindcss"`
- ✅ Replaced `@layer base` with `@theme` directive
- ✅ Changed HSL values (`0 0% 100%`) to direct hex colors (`#ffffff`)
- ✅ Prefixed all custom properties with `--color-` for clarity
- ✅ Removed `@apply` directives in favor of direct CSS properties
- ✅ Kept blue (#3b82f6) and olive (#a3e635/#65a30d) color scheme

### 2. Updated `tailwind.config.js`

**Before** (v3 syntax):
```javascript
colors: {
  border: "hsl(var(--border))",
  input: "hsl(var(--input))",
  ring: "hsl(var(--ring))",
  background: "hsl(var(--background))",
  foreground: "hsl(var(--foreground))",
  // ...
}
```

**After** (v4 syntax):
```javascript
colors: {
  border: "var(--color-border)",
  input: "var(--color-input)",
  ring: "var(--color-ring)",
  background: "var(--color-background)",
  foreground: "var(--color-foreground)",
  // ...
}
```

**Key Changes**:
- ✅ Removed `hsl()` wrapper
- ✅ Updated variable names to match new `--color-*` prefix
- ✅ Updated border radius to use `--radius-*` variables

### 3. Cleared Vite Cache

```bash
rm -rf node_modules/.vite
npm run dev
```

Vite caches compiled CSS, so clearing the cache was necessary to pick up the changes.

## Verification

After applying the fix:

1. ✅ Dev server starts without errors
2. ✅ No PostCSS/Tailwind errors in console
3. ✅ No TypeScript errors
4. ✅ UI loads successfully at http://localhost:5173
5. ✅ Blue and olive color scheme preserved
6. ✅ Dark mode still works correctly
7. ✅ All components render properly

## Tailwind CSS 4.0 Migration Summary

### What Changed in Tailwind v4

| Feature | Tailwind v3 | Tailwind v4 |
|---------|-------------|-------------|
| **Import** | `@tailwind base/components/utilities` | `@import "tailwindcss"` |
| **Theme** | `@layer base` with CSS vars | `@theme` directive |
| **Colors** | HSL values (`0 0% 100%`) | Direct colors (`#ffffff`) |
| **Custom Props** | `--border`, `--primary` | `--color-border`, `--color-primary` |
| **Config** | `hsl(var(--border))` | `var(--color-border)` |
| **PostCSS** | `tailwindcss: {}` | `'@tailwindcss/postcss': {}` |

### Benefits of v4 Syntax

1. **Simpler**: No more HSL conversion, direct color values
2. **Clearer**: `--color-*` prefix makes purpose obvious
3. **Faster**: Better performance with new engine
4. **Modern**: Uses latest CSS features
5. **Type-safe**: Better IDE support

## Files Modified

1. `src/index.css` - Complete rewrite for v4 syntax
2. `tailwind.config.js` - Updated color references
3. `node_modules/.vite/` - Cleared cache (not tracked)

## Color Scheme Preserved

The blue and olive color scheme was maintained:

**Light Mode**:
- Primary: `#3b82f6` (Blue-500)
- Secondary: `#a3e635` (Lime-400 / Olive)

**Dark Mode**:
- Primary: `#3b82f6` (Blue-500)
- Secondary: `#65a30d` (Lime-600 / Olive)

## Next Steps

The Tailwind CSS 4.0 configuration is now correct and working. All existing components will continue to work as expected because:

1. The Tailwind config still exposes the same color names (`primary`, `secondary`, etc.)
2. The CSS custom properties are still available (just with new names)
3. All Tailwind utility classes work the same way

**No component changes required!** 🎉

## References

- [Tailwind CSS v4 Documentation](https://tailwindcss.com/docs)
- [Tailwind CSS v4 Migration Guide](https://tailwindcss.com/docs/upgrade-guide)
- [@tailwindcss/postcss Plugin](https://tailwindcss.com/docs/installation/using-postcss)
- [@theme Directive](https://tailwindcss.com/docs/functions-and-directives#theme)

---

**Status**: ✅ **FIXED** - Tailwind CSS 4.0 now working correctly with CPAS4 UI

