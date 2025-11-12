/**
 * Test Page - Minimal test to verify module loading
 */

export default function TestPage() {
  console.log('[TestPage] Component loaded successfully!');
  
  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold">Test Page</h1>
      <p className="mt-4">If you can see this, the page loaded successfully!</p>
    </div>
  );
}
