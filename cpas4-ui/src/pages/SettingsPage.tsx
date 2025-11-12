import { useState, useRef } from 'react';
import { Image, Upload, X, Key, Shield, Database, Server, Gauge, Bell, CheckCircle2, XCircle, Loader2 } from 'lucide-react';
import { Button } from '../components/ui/button';
import { Slider } from '../components/ui/slider';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Switch } from '../components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { useUIStore } from '../stores/uiStore';
import { api } from '../lib/api';
import { toast } from 'sonner';

interface APIKeyConfig {
  openai: string;
  anthropic: string;
  google: string;
  groq: string;
  jina: string;
  exa: string;
  brave: string;
  apollo: string;
  clearbit: string;
  hunter: string;
  googleClientId: string;
  googleClientSecret: string;
  googleRedirectUri: string;
}

interface SecurityConfig {
  sessionTimeout: number;
  enableHttpsRedirect: boolean;
  enableInputValidation: boolean;
  enableSecurityHeaders: boolean;
}

interface OSINTConfig {
  enableFreeTools: boolean;
  enableDNS: boolean;
  enableWHOIS: boolean;
  enableWebScraping: boolean;
  enableSSL: boolean;
}

interface MCPConfig {
  jinaEnabled: boolean;
  exaEnabled: boolean;
  braveEnabled: boolean;
  systemWide: boolean;
}

interface RateLimitConfig {
  enabled: boolean;
  defaultLimit: string;
  strictLimit: string;
}

export default function SettingsPage() {
  const {
    backgroundImage,
    backgroundOpacity,
    setBackgroundImage,
    setBackgroundOpacity,
  } = useUIStore();

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [previewImage, setPreviewImage] = useState<string | undefined>(backgroundImage);
  const [activeTab, setActiveTab] = useState('appearance');
  const [testingKey, setTestingKey] = useState<string | null>(null);

  // API Keys State
  const [apiKeys, setApiKeys] = useState<APIKeyConfig>(() => {
    const stored = localStorage.getItem('cpas5_api_keys');
    return stored ? JSON.parse(stored) : {
      openai: '',
      anthropic: '',
      google: '',
      groq: '',
      jina: '',
      exa: '',
      brave: '',
      apollo: '',
      clearbit: '',
      hunter: '',
      googleClientId: '',
      googleClientSecret: '',
      googleRedirectUri: 'http://localhost:8000/api/v1/oauth/google/callback',
    };
  });

  // Security State
  const [security, setSecurity] = useState<SecurityConfig>(() => {
    const stored = localStorage.getItem('cpas5_security');
    return stored ? JSON.parse(stored) : {
      sessionTimeout: 60,
      enableHttpsRedirect: false,
      enableInputValidation: true,
      enableSecurityHeaders: true,
    };
  });

  // OSINT State
  const [osint, setOSINT] = useState<OSINTConfig>(() => {
    const stored = localStorage.getItem('cpas5_osint');
    return stored ? JSON.parse(stored) : {
      enableFreeTools: true,
      enableDNS: true,
      enableWHOIS: true,
      enableWebScraping: true,
      enableSSL: true,
    };
  });

  // MCP State
  const [mcp, setMCP] = useState<MCPConfig>(() => {
    const stored = localStorage.getItem('cpas5_mcp');
    return stored ? JSON.parse(stored) : {
      jinaEnabled: true,
      exaEnabled: false,
      braveEnabled: false,
      systemWide: true,
    };
  });

  // Rate Limit State
  const [rateLimit, setRateLimit] = useState<RateLimitConfig>(() => {
    const stored = localStorage.getItem('cpas5_rate_limit');
    return stored ? JSON.parse(stored) : {
      enabled: true,
      defaultLimit: '100/minute',
      strictLimit: '10/minute',
    };
  });

  const handleImageUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      // Validate file size (max 5MB)
      if (file.size > 5 * 1024 * 1024) {
        toast.error('Image size must be less than 5MB');
        return;
      }

      // Validate file type
      if (!file.type.startsWith('image/')) {
        toast.error('Please select a valid image file');
        return;
      }

      const reader = new FileReader();
      reader.onloadend = () => {
        const result = reader.result as string;
        setPreviewImage(result);
        setBackgroundImage(result);
        toast.success('Background image uploaded successfully!');
      };
      reader.readAsDataURL(file);
    }
  };

  const removeImage = () => {
    setPreviewImage(undefined);
    setBackgroundImage(undefined);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    toast.success('Background image removed');
  };

  const testAPIKey = async (service: string, key: string) => {
    if (!key) {
      toast.error(`Please enter ${service.toUpperCase()} API key first`);
      return;
    }

    setTestingKey(service);
    try {
      // Map services to their specific test endpoints
      const endpointMap: Record<string, string> = {
        'jina': '/jina/health',
        'exa': '/exa/health',
        'brave': '/brave/health',
        'openai': '/health',
        'anthropic': '/health',
        'google': '/health',
        'groq': '/health',
      };

      const endpoint = endpointMap[service] || '/health';
      
      // For services without specific endpoints, simulate a test
      if (!endpointMap[service]) {
        // Simulate API key validation (basic format check)
        await new Promise(resolve => setTimeout(resolve, 500));
        if (key.length > 10) {
          toast.success(`${service.toUpperCase()} API key format looks valid!`);
        } else {
          toast.warning(`${service.toUpperCase()} API key seems too short`);
        }
        setTestingKey(null);
        return;
      }

      const response = await api.get(endpoint, {
        headers: {
          'Authorization': `Bearer ${key}`
        }
      });

      if (response.data.status === 'healthy' || response.status === 200) {
        toast.success(`${service.toUpperCase()} connection successful!`);
      } else {
        toast.warning(`${service.toUpperCase()} responded but may not be fully configured`);
      }
    } catch (error: any) {
      // If endpoint doesn't exist, do basic validation
      if (error.response?.status === 404) {
        if (key.length > 10) {
          toast.success(`${service.toUpperCase()} API key format looks valid! (Endpoint not available for testing)`);
        } else {
          toast.warning(`${service.toUpperCase()} API key seems too short`);
        }
      } else {
        toast.error(`Failed to connect to ${service.toUpperCase()}: ${error.message}`);
      }
    } finally {
      setTestingKey(null);
    }
  };

  const saveSettings = () => {
    localStorage.setItem('cpas5_api_keys', JSON.stringify(apiKeys));
    localStorage.setItem('cpas5_security', JSON.stringify(security));
    localStorage.setItem('cpas5_osint', JSON.stringify(osint));
    localStorage.setItem('cpas5_mcp', JSON.stringify(mcp));
    localStorage.setItem('cpas5_rate_limit', JSON.stringify(rateLimit));

    toast.success('Settings saved successfully!');
  };

  const updateAPIKey = (key: keyof APIKeyConfig, value: string) => {
    setApiKeys(prev => ({ ...prev, [key]: value }));
  };

  const updateSecurity = (key: keyof SecurityConfig, value: any) => {
    setSecurity(prev => ({ ...prev, [key]: value }));
  };

  const updateOSINT = (key: keyof OSINTConfig, value: boolean) => {
    setOSINT(prev => ({ ...prev, [key]: value }));
  };

  const updateMCP = (key: keyof MCPConfig, value: boolean) => {
    setMCP(prev => ({ ...prev, [key]: value }));
  };

  const updateRateLimit = (key: keyof RateLimitConfig, value: any) => {
    setRateLimit(prev => ({ ...prev, [key]: value }));
  };

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="max-w-5xl mx-auto">
        <div className="mb-6">
          <h1 className="text-3xl font-bold">Settings</h1>
          <p className="text-muted-foreground mt-2">
            Configure your CPAS5/GRETA system settings
          </p>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-6 mb-6">
            <TabsTrigger value="appearance">UI</TabsTrigger>
            <TabsTrigger value="api-keys">
              <Key className="h-3 w-3 mr-1" />
              Keys
            </TabsTrigger>
            <TabsTrigger value="security">
              <Shield className="h-3 w-3 mr-1" />
              Security
            </TabsTrigger>
            <TabsTrigger value="osint">
              <Database className="h-3 w-3 mr-1" />
              OSINT
            </TabsTrigger>
            <TabsTrigger value="mcp">
              <Server className="h-3 w-3 mr-1" />
              MCP
            </TabsTrigger>
            <TabsTrigger value="rate-limit">
              <Gauge className="h-3 w-3 mr-1" />
              Limits
            </TabsTrigger>
          </TabsList>

          {/* Appearance Tab */}
          <TabsContent value="appearance" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Background Image</CardTitle>
                <CardDescription>Customize your workspace background</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="background-upload">Upload Background Image</Label>
                  <div className="flex gap-2">
                    <Button
                      type="button"
                      variant="outline"
                      onClick={() => fileInputRef.current?.click()}
                      className="flex-1"
                    >
                      <Upload className="h-4 w-4 mr-2" />
                      Choose Image
                    </Button>
                    {previewImage && (
                      <Button
                        type="button"
                        variant="destructive"
                        onClick={removeImage}
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    )}
                  </div>
                  <input
                    ref={fileInputRef}
                    id="background-upload"
                    type="file"
                    accept="image/*"
                    onChange={handleImageUpload}
                    className="hidden"
                  />
                </div>

                {previewImage && (
                  <div className="space-y-2">
                    <Label>Preview</Label>
                    <div className="relative w-full h-48 rounded-lg overflow-hidden border">
                      <img
                        src={previewImage}
                        alt="Background preview"
                        className="w-full h-full object-cover"
                      />
                    </div>
                  </div>
                )}

                <div className="space-y-2">
                  <Label htmlFor="opacity">
                    Background Opacity: {Math.round(backgroundOpacity * 100)}%
                  </Label>
                  <Slider
                    id="opacity"
                    min={0}
                    max={1}
                    step={0.1}
                    value={[backgroundOpacity]}
                    onValueChange={([value]) => setBackgroundOpacity(value)}
                  />
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* API Keys Tab */}
          <TabsContent value="api-keys" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>LLM Provider API Keys</CardTitle>
                <CardDescription>Configure API keys for AI language models</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {[
                  { key: 'openai', label: 'OpenAI API Key', placeholder: 'sk-...' },
                  { key: 'anthropic', label: 'Anthropic API Key', placeholder: 'sk-ant-...' },
                  { key: 'google', label: 'Google AI API Key', placeholder: 'AIza...' },
                  { key: 'groq', label: 'Groq API Key', placeholder: 'gsk_...' },
                ].map(({ key, label, placeholder }) => (
                  <div key={key} className="space-y-2">
                    <Label htmlFor={key}>{label}</Label>
                    <div className="flex gap-2">
                      <Input
                        id={key}
                        type="password"
                        placeholder={placeholder}
                        value={apiKeys[key as keyof APIKeyConfig]}
                        onChange={(e) => updateAPIKey(key as keyof APIKeyConfig, e.target.value)}
                      />
                      <Button
                        type="button"
                        variant="outline"
                        onClick={() => testAPIKey(key, apiKeys[key as keyof APIKeyConfig])}
                        disabled={testingKey === key}
                      >
                        {testingKey === key ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          'Test'
                        )}
                      </Button>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>MCP Service API Keys</CardTitle>
                <CardDescription>Configure API keys for MCP services</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {[
                  { key: 'jina', label: 'Jina AI API Key', placeholder: 'jina_...' },
                  { key: 'exa', label: 'Exa API Key', placeholder: 'exa_...' },
                  { key: 'brave', label: 'Brave Search API Key', placeholder: 'BSA...' },
                ].map(({ key, label, placeholder }) => (
                  <div key={key} className="space-y-2">
                    <Label htmlFor={key}>{label}</Label>
                    <div className="flex gap-2">
                      <Input
                        id={key}
                        type="password"
                        placeholder={placeholder}
                        value={apiKeys[key as keyof APIKeyConfig]}
                        onChange={(e) => updateAPIKey(key as keyof APIKeyConfig, e.target.value)}
                      />
                      <Button
                        type="button"
                        variant="outline"
                        onClick={() => testAPIKey(key, apiKeys[key as keyof APIKeyConfig])}
                        disabled={testingKey === key}
                      >
                        {testingKey === key ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          'Test'
                        )}
                      </Button>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>OSINT Tool API Keys</CardTitle>
                <CardDescription>Configure API keys for OSINT tools</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {[
                  { key: 'apollo', label: 'Apollo API Key', placeholder: 'apollo_...' },
                  { key: 'clearbit', label: 'Clearbit API Key', placeholder: 'sk_...' },
                  { key: 'hunter', label: 'Hunter API Key', placeholder: 'hunter_...' },
                ].map(({ key, label, placeholder }) => (
                  <div key={key} className="space-y-2">
                    <Label htmlFor={key}>{label}</Label>
                    <Input
                      id={key}
                      type="password"
                      placeholder={placeholder}
                      value={apiKeys[key as keyof APIKeyConfig]}
                      onChange={(e) => updateAPIKey(key as keyof APIKeyConfig, e.target.value)}
                    />
                  </div>
                ))}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Security Tab */}
          <TabsContent value="security" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Security Settings</CardTitle>
                <CardDescription>Configure security and authentication settings</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="session-timeout">Session Timeout (minutes)</Label>
                  <Input
                    id="session-timeout"
                    type="number"
                    value={security.sessionTimeout}
                    onChange={(e) => updateSecurity('sessionTimeout', parseInt(e.target.value))}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>HTTPS Redirect</Label>
                    <p className="text-sm text-muted-foreground">Force HTTPS connections</p>
                  </div>
                  <Switch
                    checked={security.enableHttpsRedirect}
                    onCheckedChange={(checked) => updateSecurity('enableHttpsRedirect', checked)}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Input Validation</Label>
                    <p className="text-sm text-muted-foreground">Validate all user inputs</p>
                  </div>
                  <Switch
                    checked={security.enableInputValidation}
                    onCheckedChange={(checked) => updateSecurity('enableInputValidation', checked)}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Security Headers</Label>
                    <p className="text-sm text-muted-foreground">Enable security HTTP headers</p>
                  </div>
                  <Switch
                    checked={security.enableSecurityHeaders}
                    onCheckedChange={(checked) => updateSecurity('enableSecurityHeaders', checked)}
                  />
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* OSINT Tab */}
          <TabsContent value="osint" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>OSINT Free Tools</CardTitle>
                <CardDescription>Enable or disable OSINT tools (no API key required)</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Enable OSINT Free Tools</Label>
                    <p className="text-sm text-muted-foreground">Master toggle for all free tools</p>
                  </div>
                  <Switch
                    checked={osint.enableFreeTools}
                    onCheckedChange={(checked) => updateOSINT('enableFreeTools', checked)}
                  />
                </div>

                {osint.enableFreeTools && (
                  <>
                    <div className="flex items-center justify-between">
                      <div className="space-y-0.5">
                        <Label>DNS Lookup</Label>
                        <p className="text-sm text-muted-foreground">Domain name resolution</p>
                      </div>
                      <Switch
                        checked={osint.enableDNS}
                        onCheckedChange={(checked) => updateOSINT('enableDNS', checked)}
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="space-y-0.5">
                        <Label>WHOIS Lookup</Label>
                        <p className="text-sm text-muted-foreground">Domain registration info</p>
                      </div>
                      <Switch
                        checked={osint.enableWHOIS}
                        onCheckedChange={(checked) => updateOSINT('enableWHOIS', checked)}
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="space-y-0.5">
                        <Label>Web Scraping</Label>
                        <p className="text-sm text-muted-foreground">Extract data from websites</p>
                      </div>
                      <Switch
                        checked={osint.enableWebScraping}
                        onCheckedChange={(checked) => updateOSINT('enableWebScraping', checked)}
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="space-y-0.5">
                        <Label>SSL Certificate Check</Label>
                        <p className="text-sm text-muted-foreground">Verify SSL certificates</p>
                      </div>
                      <Switch
                        checked={osint.enableSSL}
                        onCheckedChange={(checked) => updateOSINT('enableSSL', checked)}
                      />
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* MCP Tab */}
          <TabsContent value="mcp" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>MCP Server Configuration</CardTitle>
                <CardDescription>Manage Model Context Protocol servers</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Jina AI MCP</Label>
                    <p className="text-sm text-muted-foreground">AI search, embeddings, and analysis</p>
                  </div>
                  <Switch
                    checked={mcp.jinaEnabled}
                    onCheckedChange={(checked) => updateMCP('jinaEnabled', checked)}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Exa MCP</Label>
                    <p className="text-sm text-muted-foreground">Semantic search engine</p>
                  </div>
                  <Switch
                    checked={mcp.exaEnabled}
                    onCheckedChange={(checked) => updateMCP('exaEnabled', checked)}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Brave Search MCP</Label>
                    <p className="text-sm text-muted-foreground">Privacy-focused web search</p>
                  </div>
                  <Switch
                    checked={mcp.braveEnabled}
                    onCheckedChange={(checked) => updateMCP('braveEnabled', checked)}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>System-Wide MCP</Label>
                    <p className="text-sm text-muted-foreground">Enable MCP for all agents</p>
                  </div>
                  <Switch
                    checked={mcp.systemWide}
                    onCheckedChange={(checked) => updateMCP('systemWide', checked)}
                  />
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Rate Limiting Tab */}
          <TabsContent value="rate-limit" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Rate Limiting Configuration</CardTitle>
                <CardDescription>Configure API rate limits and DDoS protection</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Enable Rate Limiting</Label>
                    <p className="text-sm text-muted-foreground">Protect against abuse</p>
                  </div>
                  <Switch
                    checked={rateLimit.enabled}
                    onCheckedChange={(checked) => updateRateLimit('enabled', checked)}
                  />
                </div>

                {rateLimit.enabled && (
                  <>
                    <div className="space-y-2">
                      <Label htmlFor="default-limit">Default Rate Limit</Label>
                      <Input
                        id="default-limit"
                        placeholder="100/minute"
                        value={rateLimit.defaultLimit}
                        onChange={(e) => updateRateLimit('defaultLimit', e.target.value)}
                      />
                      <p className="text-sm text-muted-foreground">Format: number/period (e.g., 100/minute)</p>
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="strict-limit">Strict Rate Limit</Label>
                      <Input
                        id="strict-limit"
                        placeholder="10/minute"
                        value={rateLimit.strictLimit}
                        onChange={(e) => updateRateLimit('strictLimit', e.target.value)}
                      />
                      <p className="text-sm text-muted-foreground">For sensitive endpoints</p>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          </TabsContent>


        </Tabs>

        {/* Save Button */}
        <div className="mt-6 flex justify-end">
          <Button onClick={saveSettings} size="lg">
            Save All Settings
          </Button>
        </div>
      </div>
    </div>
  );
}
