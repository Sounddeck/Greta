import { useState } from 'react';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Card } from '../ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { Loader2, Link as LinkIcon, Code, Download, Copy, CheckCircle2, AlertCircle } from 'lucide-react';
import { toast } from 'sonner';

interface GeneratedCodeResponse {
  code: string;
  styles?: string;
  framework: string;
  component_name: string;
  file_id: string;
  node_id?: string;
  generated_at: string;
  dependencies: string[];
  instructions?: string;
}

export const FigmaIntegration = () => {
  const [figmaUrl, setFigmaUrl] = useState('');
  const [framework, setFramework] = useState('react');
  const [componentName, setComponentName] = useState('Component');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedCode, setGeneratedCode] = useState<GeneratedCodeResponse | null>(null);
  const [copiedCode, setCopiedCode] = useState(false);
  const [copiedStyles, setCopiedStyles] = useState(false);

  const handleGenerate = async () => {
    if (!figmaUrl) {
      toast.error('Please enter a Figma URL');
      return;
    }

    if (!componentName || componentName.trim() === '') {
      toast.error('Please enter a component name');
      return;
    }

    setIsGenerating(true);
    setGeneratedCode(null);

    try {
      const response = await fetch('/api/v1/figma/generate-code', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          figma_url: figmaUrl,
          framework,
          component_name: componentName,
          use_cache: true,
          include_styles: true,
          responsive: true,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to generate code');
      }

      const data: GeneratedCodeResponse = await response.json();
      setGeneratedCode(data);
      toast.success('Code generated successfully!');
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to generate code from Figma';
      toast.error(errorMessage);
      console.error(error);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleCopyCode = async () => {
    if (generatedCode?.code) {
      await navigator.clipboard.writeText(generatedCode.code);
      setCopiedCode(true);
      toast.success('Code copied to clipboard!');
      setTimeout(() => setCopiedCode(false), 2000);
    }
  };

  const handleCopyStyles = async () => {
    if (generatedCode?.styles) {
      await navigator.clipboard.writeText(generatedCode.styles);
      setCopiedStyles(true);
      toast.success('Styles copied to clipboard!');
      setTimeout(() => setCopiedStyles(false), 2000);
    }
  };

  const handleDownload = () => {
    if (!generatedCode) return;

    const fileExtensions: Record<string, string> = {
      react: '.jsx',
      vue: '.vue',
      svelte: '.svelte',
      html: '.html',
      angular: '.component.ts',
      next: '.tsx',
    };

    const ext = fileExtensions[generatedCode.framework] || '.txt';
    const fileName = `${generatedCode.component_name}${ext}`;

    // Download code file
    const codeBlob = new Blob([generatedCode.code], { type: 'text/plain' });
    const codeUrl = URL.createObjectURL(codeBlob);
    const codeLink = document.createElement('a');
    codeLink.href = codeUrl;
    codeLink.download = fileName;
    codeLink.click();
    URL.revokeObjectURL(codeUrl);

    // Download styles file if present
    if (generatedCode.styles) {
      const stylesBlob = new Blob([generatedCode.styles], { type: 'text/plain' });
      const stylesUrl = URL.createObjectURL(stylesBlob);
      const stylesLink = document.createElement('a');
      stylesLink.href = stylesUrl;
      stylesLink.download = `${generatedCode.component_name}.css`;
      stylesLink.click();
      URL.revokeObjectURL(stylesUrl);
    }

    toast.success('Files downloaded!');
  };

  return (
    <div className="space-y-6">
      <Card className="p-6">
        <div className="space-y-4">
          <div>
            <h2 className="text-2xl font-bold mb-2">Figma to Code</h2>
            <p className="text-sm text-gray-600">
              Convert your Figma designs into production-ready code. Paste a Figma link and generate
              code in your preferred framework.
            </p>
          </div>

          <div className="space-y-4">
            <div>
              <Label htmlFor="figma-url" className="text-sm font-medium mb-2 block">
                Figma URL
              </Label>
              <div className="relative">
                <LinkIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                <Input
                  id="figma-url"
                  type="url"
                  placeholder="https://www.figma.com/file/..."
                  value={figmaUrl}
                  onChange={(e) => setFigmaUrl(e.target.value)}
                  className="pl-10"
                />
              </div>
              <p className="text-xs text-gray-500 mt-1">
                Paste a link to a Figma file or specific frame
              </p>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="framework" className="text-sm font-medium mb-2 block">
                  Framework
                </Label>
                <Select value={framework} onValueChange={setFramework}>
                  <SelectTrigger id="framework">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="react">React</SelectItem>
                    <SelectItem value="vue">Vue</SelectItem>
                    <SelectItem value="svelte">Svelte</SelectItem>
                    <SelectItem value="html">HTML/CSS</SelectItem>
                    <SelectItem value="angular" disabled>Angular (Coming Soon)</SelectItem>
                    <SelectItem value="next" disabled>Next.js (Coming Soon)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label htmlFor="component-name" className="text-sm font-medium mb-2 block">
                  Component Name
                </Label>
                <Input
                  id="component-name"
                  type="text"
                  placeholder="Component"
                  value={componentName}
                  onChange={(e) => setComponentName(e.target.value)}
                />
              </div>
            </div>

            <Button
              onClick={handleGenerate}
              disabled={isGenerating || !figmaUrl}
              className="w-full"
              size="lg"
            >
              {isGenerating ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Generating Code...
                </>
              ) : (
                <>
                  <Code className="w-4 h-4 mr-2" />
                  Generate Code
                </>
              )}
            </Button>
          </div>
        </div>
      </Card>

      {generatedCode && (
        <Card className="p-6">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold">Generated Code</h3>
                <p className="text-sm text-gray-600">
                  {generatedCode.framework.toUpperCase()} • {generatedCode.component_name}
                </p>
              </div>
              <div className="flex gap-2">
                <Button onClick={handleCopyCode} variant="outline" size="sm">
                  {copiedCode ? (
                    <>
                      <CheckCircle2 className="w-4 h-4 mr-2" />
                      Copied!
                    </>
                  ) : (
                    <>
                      <Copy className="w-4 h-4 mr-2" />
                      Copy Code
                    </>
                  )}
                </Button>
                {generatedCode.styles && (
                  <Button onClick={handleCopyStyles} variant="outline" size="sm">
                    {copiedStyles ? (
                      <>
                        <CheckCircle2 className="w-4 h-4 mr-2" />
                        Copied!
                      </>
                    ) : (
                      <>
                        <Copy className="w-4 h-4 mr-2" />
                        Copy Styles
                      </>
                    )}
                  </Button>
                )}
                <Button onClick={handleDownload} variant="outline" size="sm">
                  <Download className="w-4 h-4 mr-2" />
                  Download
                </Button>
              </div>
            </div>

            <div>
              <Label className="text-sm font-medium mb-2 block">Component Code</Label>
              <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto text-sm">
                <code>{generatedCode.code}</code>
              </pre>
            </div>

            {generatedCode.styles && (
              <div>
                <Label className="text-sm font-medium mb-2 block">Styles (CSS)</Label>
                <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto text-sm">
                  <code>{generatedCode.styles}</code>
                </pre>
              </div>
            )}

            {generatedCode.dependencies.length > 0 && (
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <div className="flex items-start gap-2">
                  <AlertCircle className="w-5 h-5 text-blue-600 mt-0.5" />
                  <div>
                    <h4 className="font-medium text-blue-900">Dependencies Required</h4>
                    <p className="text-sm text-blue-700 mt-1">
                      Install: <code className="bg-blue-100 px-2 py-0.5 rounded">
                        npm install {generatedCode.dependencies.join(' ')}
                      </code>
                    </p>
                  </div>
                </div>
              </div>
            )}

            {generatedCode.instructions && (
              <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
                <h4 className="font-medium text-gray-900 mb-2">Setup Instructions</h4>
                <p className="text-sm text-gray-700">{generatedCode.instructions}</p>
              </div>
            )}

            <div className="text-xs text-gray-500">
              Generated at {new Date(generatedCode.generated_at).toLocaleString()} • File ID: {generatedCode.file_id}
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};

export default FigmaIntegration;

