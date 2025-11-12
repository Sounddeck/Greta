import { useState, useRef } from 'react';
import { Image, Upload, X, Key, Shield, Database, Server, Gauge, Bell, CheckCircle2, XCircle, Loader2 } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '../ui/dialog';
import { Button } from '../ui/button';
import { Slider } from '../ui/slider';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Switch } from '../ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { useUIStore } from '../../stores/uiStore';
import { api } from '../../lib/api';
import { toast } from 'sonner';

interface SettingsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

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

export function SettingsDialog({ open, onOpenChange }: SettingsDialogProps) {
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

  // Rate Limiting State
  const [rateLimit, setRateLimit] = useState<RateLimitConfig>(() => {
    const stored = localStorage.getItem('cpas5_rate_limit');
    return stored ? JSON.parse(stored) : {
      enabled: true,
      defaultLimit: '100/minute',
      strictLimit: '10/minute',
    };
  });

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const result = e.target?.result as string;
        setPreviewImage(result);
        setBackgroundImage(result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleRemoveBackground = () => {
    setPreviewImage(undefined);
    setBackgroundImage(undefined);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const testAPIKey = async (service: string, key: string) => {
    if (!key) {
      toast.error('Please enter an API key first');
      return;
    }

    setTestingKey(service);
    try {
      // Test the API key based on service
      let endpoint = '';
      switch (service) {
        case 'jina':
          endpoint = '/jina/health';
          break;
        case 'osint_free':
          endpoint = '/osint/free/health';
          break;
        default:
          toast.info('Test connection not implemented for this service yet');
          setTestingKey(null);
          return;
      }

      const response = await api.get(endpoint);
      if (response.data.status === 'healthy') {
        toast.success(`${service.toUpperCase()} connection successful!`);
      } else {
        toast.warning(`${service.toUpperCase()} responded but may not be fully configured`);
      }
    } catch (error: any) {
      toast.error(`Failed to connect to ${service.toUpperCase()}: ${error.message}`);
    } finally {
      setTestingKey(null);
    }
  };

  const saveSettings = () => {
    // Save to localStorage
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
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[700px] max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Settings</DialogTitle>
          <DialogDescription>
            Configure your CPAS5/GRETA system settings
          </DialogDescription>
        </DialogHeader>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-7">
            <TabsTrigger value="appearance">UI</TabsTrigger>
            <TabsTrigger value="api-keys">
              <Key className="h-3 w-3 mr-1" />
              API Keys
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
            <TabsTrigger value="notifications">
              <Bell className="h-3 w-3 mr-1" />
              Alerts
            </TabsTrigger>
          </TabsList>

          {/* Appearance Tab */}
          <TabsContent value="appearance" className="space-y-4 mt-4">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Image className="h-4 w-4 text-primary" />
                  <h3 className="font-medium text-sm">Background Image</h3>
                </div>
                {previewImage && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleRemoveBackground}
                    className="h-8 text-xs"
                  >
                    <X className="h-3 w-3 mr-1" />
                    Remove
                  </Button>
                )}
              </div>

              {previewImage && (
                <div className="relative w-full h-32 rounded-md overflow-hidden border border-border">
                  <img
                    src={previewImage}
                    alt="Background preview"
                    className="w-full h-full object-cover"
                    style={{ opacity: backgroundOpacity }}
                  />
                </div>
              )}

              <div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  className="hidden"
                />
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => fileInputRef.current?.click()}
                  className="w-full"
                >
                  <Upload className="h-4 w-4 mr-2" />
                  {previewImage ? 'Change Background' : 'Upload Background'}
                </Button>
              </div>

              <p className="text-xs text-muted-foreground">
                Supported formats: JPG, PNG, GIF. Recommended size: 1920x1080px
              </p>
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="font-medium text-sm">Background Opacity</h3>
                <span className="text-sm text-muted-foreground">
                  {Math.round(backgroundOpacity * 100)}%
                </span>
              </div>

              <Slider
                value={[backgroundOpacity]}
                onValueChange={(value) => setBackgroundOpacity(value[0])}
                min={0}
                max={1}
                step={0.05}
                className="w-full"
              />

              <p className="text-xs text-muted-foreground">
                Adjust the opacity of the background image overlay
              </p>
            </div>
          </TabsContent>

          {/* API Keys Tab */}
          <TabsContent value="api-keys" className="space-y-4 mt-4">
            <div className="space-y-4">
              <div>
                <h3 className="font-medium text-sm mb-3">LLM Providers</h3>
                <div className="space-y-3">
                  <div className="space-y-1.5">
                    <Label htmlFor="openai-key" className="text-xs">OpenAI API Key</Label>
                    <div className="flex gap-2">
                      <Input
                        id="openai-key"
                        type="password"
                        placeholder="sk-..."
                        value={apiKeys.openai}
                        onChange={(e) => updateAPIKey('openai', e.target.value)}
                        className="font-mono text-xs"
                      />
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => testAPIKey('openai', apiKeys.openai)}
                        disabled={testingKey === 'openai'}
                      >
                        {testingKey === 'openai' ? <Loader2 className="h-3 w-3 animate-spin" /> : 'Test'}
                      </Button>
                    </div>
                  </div>

                  <div className="space-y-1.5">
                    <Label htmlFor="anthropic-key" className="text-xs">Anthropic API Key</Label>
                    <div className="flex gap-2">
                      <Input
                        id="anthropic-key"
                        type="password"
                        placeholder="sk-ant-..."
                        value={apiKeys.anthropic}
                        onChange={(e) => updateAPIKey('anthropic', e.target.value)}
                        className="font-mono text-xs"
                      />
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => testAPIKey('anthropic', apiKeys.anthropic)}
                        disabled={testingKey === 'anthropic'}
                      >
                        {testingKey === 'anthropic' ? <Loader2 className="h-3 w-3 animate-spin" /> : 'Test'}
                      </Button>
                    </div>
                  </div>

                  <div className="space-y-1.5">
                    <Label htmlFor="google-key" className="text-xs">Google AI API Key</Label>
                    <div className="flex gap-2">
                      <Input
                        id="google-key"
                        type="password"
                        placeholder="AIza..."
                        value={apiKeys.google}
                        onChange={(e) => updateAPIKey('google', e.target.value)}
                        className="font-mono text-xs"
                      />
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => testAPIKey('google', apiKeys.google)}
                        disabled={testingKey === 'google'}
                      >
                        {testingKey === 'google' ? <Loader2 className="h-3 w-3 animate-spin" /> : 'Test'}
                      </Button>
                    </div>
                  </div>

                  <div className="space-y-1.5">
                    <Label htmlFor="groq-key" className="text-xs">Groq API Key</Label>
                    <div className="flex gap-2">
                      <Input
                        id="groq-key"
                        type="password"
                        placeholder="gsk_..."
                        value={apiKeys.groq}
                        onChange={(e) => updateAPIKey('groq', e.target.value)}
                        className="font-mono text-xs"
                      />
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => testAPIKey('groq', apiKeys.groq)}
                        disabled={testingKey === 'groq'}
                      >
                        {testingKey === 'groq' ? <Loader2 className="h-3 w-3 animate-spin" /> : 'Test'}
                      </Button>
                    </div>
                  </div>
                </div>
              </div>

              <div className="border-t pt-4">
                <h3 className="font-medium text-sm mb-3">MCP Services</h3>
                <div className="space-y-3">
                  <div className="space-y-1.5">
                    <Label htmlFor="jina-key" className="text-xs">Jina AI API Key</Label>
                    <div className="flex gap-2">
                      <Input
                        id="jina-key"
                        type="password"
                        placeholder="jina_..."
                        value={apiKeys.jina}
                        onChange={(e) => updateAPIKey('jina', e.target.value)}
                        className="font-mono text-xs"
                      />
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => testAPIKey('jina', apiKeys.jina)}
                        disabled={testingKey === 'jina'}
                      >
                        {testingKey === 'jina' ? <Loader2 className="h-3 w-3 animate-spin" /> : 'Test'}
                      </Button>
                    </div>
                  </div>

                  <div className="space-y-1.5">
                    <Label htmlFor="exa-key" className="text-xs">Exa AI API Key</Label>
                    <div className="flex gap-2">
                      <Input
                        id="exa-key"
                        type="password"
                        placeholder="exa_..."
                        value={apiKeys.exa}
                        onChange={(e) => updateAPIKey('exa', e.target.value)}
                        className="font-mono text-xs"
                      />
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => testAPIKey('exa', apiKeys.exa)}
                        disabled={testingKey === 'exa'}
                      >
                        {testingKey === 'exa' ? <Loader2 className="h-3 w-3 animate-spin" /> : 'Test'}
                      </Button>
                    </div>
                  </div>

                  <div className="space-y-1.5">
                    <Label htmlFor="brave-key" className="text-xs">Brave Search API Key</Label>
                    <div className="flex gap-2">
                      <Input
                        id="brave-key"
                        type="password"
                        placeholder="BSA..."
                        value={apiKeys.brave}
                        onChange={(e) => updateAPIKey('brave', e.target.value)}
                        className="font-mono text-xs"
                      />
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => testAPIKey('brave', apiKeys.brave)}
                        disabled={testingKey === 'brave'}
                      >
                        {testingKey === 'brave' ? <Loader2 className="h-3 w-3 animate-spin" /> : 'Test'}
                      </Button>
                    </div>
                  </div>
                </div>
              </div>

              <div className="border-t pt-4">
                <h3 className="font-medium text-sm mb-3">OSINT Tools</h3>
                <div className="space-y-3">
                  <div className="space-y-1.5">
                    <Label htmlFor="apollo-key" className="text-xs">Apollo.io API Key</Label>
                    <Input
                      id="apollo-key"
                      type="password"
                      placeholder="apollo_..."
                      value={apiKeys.apollo}
                      onChange={(e) => updateAPIKey('apollo', e.target.value)}
                      className="font-mono text-xs"
                    />
                  </div>

                  <div className="space-y-1.5">
                    <Label htmlFor="clearbit-key" className="text-xs">Clearbit API Key</Label>
                    <Input
                      id="clearbit-key"
                      type="password"
                      placeholder="sk_..."
                      value={apiKeys.clearbit}
                      onChange={(e) => updateAPIKey('clearbit', e.target.value)}
                      className="font-mono text-xs"
                    />
                  </div>

                  <div className="space-y-1.5">
                    <Label htmlFor="hunter-key" className="text-xs">Hunter.io API Key</Label>
                    <Input
                      id="hunter-key"
                      type="password"
                      placeholder="hunter_..."
                      value={apiKeys.hunter}
                      onChange={(e) => updateAPIKey('hunter', e.target.value)}
                      className="font-mono text-xs"
                    />
                  </div>
                </div>
              </div>

              <div className="border-t pt-4">
                <h3 className="font-medium text-sm mb-3">Google Workspace OAuth</h3>
                <div className="space-y-3">
                  <div className="space-y-1.5">
                    <Label htmlFor="google-client-id" className="text-xs">Client ID</Label>
                    <Input
                      id="google-client-id"
                      type="text"
                      placeholder="123456789-abc.apps.googleusercontent.com"
                      value={apiKeys.googleClientId}
                      onChange={(e) => updateAPIKey('googleClientId', e.target.value)}
                      className="font-mono text-xs"
                    />
                  </div>

                  <div className="space-y-1.5">
                    <Label htmlFor="google-client-secret" className="text-xs">Client Secret</Label>
                    <Input
                      id="google-client-secret"
                      type="password"
                      placeholder="GOCSPX-..."
                      value={apiKeys.googleClientSecret}
                      onChange={(e) => updateAPIKey('googleClientSecret', e.target.value)}
                      className="font-mono text-xs"
                    />
                  </div>

                  <div className="space-y-1.5">
                    <Label htmlFor="google-redirect-uri" className="text-xs">Redirect URI</Label>
                    <Input
                      id="google-redirect-uri"
                      type="text"
                      value={apiKeys.googleRedirectUri}
                      onChange={(e) => updateAPIKey('googleRedirectUri', e.target.value)}
                      className="font-mono text-xs"
                    />
                  </div>
                </div>
              </div>
            </div>
          </TabsContent>

          {/* Security Tab */}
          <TabsContent value="security" className="space-y-4 mt-4">
            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 rounded-md bg-accent/50">
                <div>
                  <p className="text-sm font-medium">JWT Secret</p>
                  <p className="text-xs text-muted-foreground">Configured in backend .env</p>
                </div>
                <CheckCircle2 className="h-4 w-4 text-green-500" />
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <Label htmlFor="session-timeout" className="text-sm">Session Timeout (minutes)</Label>
                    <p className="text-xs text-muted-foreground">Auto-logout after inactivity</p>
                  </div>
                  <Input
                    id="session-timeout"
                    type="number"
                    min="5"
                    max="1440"
                    value={security.sessionTimeout}
                    onChange={(e) => updateSecurity('sessionTimeout', parseInt(e.target.value))}
                    className="w-20 text-right"
                  />
                </div>

                <div className="flex items-center justify-between p-3 rounded-md border">
                  <div>
                    <Label className="text-sm">HTTPS Redirect</Label>
                    <p className="text-xs text-muted-foreground">Force HTTPS in production</p>
                  </div>
                  <Switch
                    checked={security.enableHttpsRedirect}
                    onCheckedChange={(checked) => updateSecurity('enableHttpsRedirect', checked)}
                  />
                </div>

                <div className="flex items-center justify-between p-3 rounded-md border">
                  <div>
                    <Label className="text-sm">Input Validation</Label>
                    <p className="text-xs text-muted-foreground">SQL injection, XSS protection</p>
                  </div>
                  <Switch
                    checked={security.enableInputValidation}
                    onCheckedChange={(checked) => updateSecurity('enableInputValidation', checked)}
                  />
                </div>

                <div className="flex items-center justify-between p-3 rounded-md border">
                  <div>
                    <Label className="text-sm">Security Headers</Label>
                    <p className="text-xs text-muted-foreground">Clickjacking, MIME sniffing protection</p>
                  </div>
                  <Switch
                    checked={security.enableSecurityHeaders}
                    onCheckedChange={(checked) => updateSecurity('enableSecurityHeaders', checked)}
                  />
                </div>
              </div>

              <div className="p-3 rounded-md bg-blue-500/10 border border-blue-500/20">
                <p className="text-xs text-blue-600 dark:text-blue-400">
                  <Shield className="h-3 w-3 inline mr-1" />
                  Security settings are enforced at the backend level. Changes require backend restart.
                </p>
              </div>
            </div>
          </TabsContent>

          {/* OSINT Tab */}
          <TabsContent value="osint" className="space-y-4 mt-4">
            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 rounded-md bg-accent/50">
                <div>
                  <p className="text-sm font-medium">OSINT Free Tools</p>
                  <p className="text-xs text-muted-foreground">No API keys required</p>
                </div>
                <div className="flex items-center gap-2">
                  {osint.enableFreeTools ? (
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                  ) : (
                    <XCircle className="h-4 w-4 text-gray-400" />
                  )}
                  <Switch
                    checked={osint.enableFreeTools}
                    onCheckedChange={(checked) => updateOSINT('enableFreeTools', checked)}
                  />
                </div>
              </div>

              {osint.enableFreeTools && (
                <div className="space-y-3 pl-4 border-l-2 border-primary/20">
                  <div className="flex items-center justify-between p-2 rounded-md border">
                    <div>
                      <Label className="text-sm">DNS Lookup</Label>
                      <p className="text-xs text-muted-foreground">Domain name resolution</p>
                    </div>
                    <Switch
                      checked={osint.enableDNS}
                      onCheckedChange={(checked) => updateOSINT('enableDNS', checked)}
                    />
                  </div>

                  <div className="flex items-center justify-between p-2 rounded-md border">
                    <div>
                      <Label className="text-sm">WHOIS Lookup</Label>
                      <p className="text-xs text-muted-foreground">Domain registration info</p>
                    </div>
                    <Switch
                      checked={osint.enableWHOIS}
                      onCheckedChange={(checked) => updateOSINT('enableWHOIS', checked)}
                    />
                  </div>

                  <div className="flex items-center justify-between p-2 rounded-md border">
                    <div>
                      <Label className="text-sm">Web Scraping</Label>
                      <p className="text-xs text-muted-foreground">Extract web content</p>
                    </div>
                    <Switch
                      checked={osint.enableWebScraping}
                      onCheckedChange={(checked) => updateOSINT('enableWebScraping', checked)}
                    />
                  </div>

                  <div className="flex items-center justify-between p-2 rounded-md border">
                    <div>
                      <Label className="text-sm">SSL Certificate Check</Label>
                      <p className="text-xs text-muted-foreground">Verify SSL certificates</p>
                    </div>
                    <Switch
                      checked={osint.enableSSL}
                      onCheckedChange={(checked) => updateOSINT('enableSSL', checked)}
                    />
                  </div>
                </div>
              )}

              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => testAPIKey('osint_free', 'test')}
                  disabled={testingKey === 'osint_free'}
                  className="flex-1"
                >
                  {testingKey === 'osint_free' ? (
                    <Loader2 className="h-3 w-3 animate-spin mr-2" />
                  ) : (
                    <Database className="h-3 w-3 mr-2" />
                  )}
                  Test OSINT Connection
                </Button>
              </div>
            </div>
          </TabsContent>

          {/* MCP Servers Tab */}
          <TabsContent value="mcp" className="space-y-4 mt-4">
            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 rounded-md bg-accent/50">
                <div>
                  <p className="text-sm font-medium">System-Wide MCP</p>
                  <p className="text-xs text-muted-foreground">Apply to all agents</p>
                </div>
                <Switch
                  checked={mcp.systemWide}
                  onCheckedChange={(checked) => updateMCP('systemWide', checked)}
                />
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 rounded-md border">
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-md bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
                      <Server className="h-5 w-5 text-white" />
                    </div>
                    <div>
                      <Label className="text-sm font-medium">Jina AI MCP</Label>
                      <p className="text-xs text-muted-foreground">Search, embeddings, content analysis</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {mcp.jinaEnabled ? (
                      <CheckCircle2 className="h-4 w-4 text-green-500" />
                    ) : (
                      <XCircle className="h-4 w-4 text-gray-400" />
                    )}
                    <Switch
                      checked={mcp.jinaEnabled}
                      onCheckedChange={(checked) => updateMCP('jinaEnabled', checked)}
                    />
                  </div>
                </div>

                <div className="flex items-center justify-between p-3 rounded-md border">
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-md bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
                      <Server className="h-5 w-5 text-white" />
                    </div>
                    <div>
                      <Label className="text-sm font-medium">Exa AI MCP</Label>
                      <p className="text-xs text-muted-foreground">Neural search engine</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {mcp.exaEnabled ? (
                      <CheckCircle2 className="h-4 w-4 text-green-500" />
                    ) : (
                      <XCircle className="h-4 w-4 text-gray-400" />
                    )}
                    <Switch
                      checked={mcp.exaEnabled}
                      onCheckedChange={(checked) => updateMCP('exaEnabled', checked)}
                    />
                  </div>
                </div>

                <div className="flex items-center justify-between p-3 rounded-md border">
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-md bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center">
                      <Server className="h-5 w-5 text-white" />
                    </div>
                    <div>
                      <Label className="text-sm font-medium">Brave Search MCP</Label>
                      <p className="text-xs text-muted-foreground">Privacy-focused search</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {mcp.braveEnabled ? (
                      <CheckCircle2 className="h-4 w-4 text-green-500" />
                    ) : (
                      <XCircle className="h-4 w-4 text-gray-400" />
                    )}
                    <Switch
                      checked={mcp.braveEnabled}
                      onCheckedChange={(checked) => updateMCP('braveEnabled', checked)}
                    />
                  </div>
                </div>
              </div>

              <div className="p-3 rounded-md bg-blue-500/10 border border-blue-500/20">
                <p className="text-xs text-blue-600 dark:text-blue-400">
                  <Server className="h-3 w-3 inline mr-1" />
                  Configure API keys in the "API Keys" tab before enabling MCP servers.
                </p>
              </div>
            </div>
          </TabsContent>

          {/* Rate Limiting Tab */}
          <TabsContent value="rate-limit" className="space-y-4 mt-4">
            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 rounded-md bg-accent/50">
                <div>
                  <p className="text-sm font-medium">Rate Limiting</p>
                  <p className="text-xs text-muted-foreground">Protect against abuse</p>
                </div>
                <Switch
                  checked={rateLimit.enabled}
                  onCheckedChange={(checked) => updateRateLimit('enabled', checked)}
                />
              </div>

              {rateLimit.enabled && (
                <div className="space-y-3">
                  <div className="space-y-1.5">
                    <Label htmlFor="default-limit" className="text-sm">Default Rate Limit</Label>
                    <Input
                      id="default-limit"
                      type="text"
                      placeholder="100/minute"
                      value={rateLimit.defaultLimit}
                      onChange={(e) => updateRateLimit('defaultLimit', e.target.value)}
                      className="font-mono text-xs"
                    />
                    <p className="text-xs text-muted-foreground">
                      Format: number/period (e.g., 100/minute, 1000/hour)
                    </p>
                  </div>

                  <div className="space-y-1.5">
                    <Label htmlFor="strict-limit" className="text-sm">Strict Endpoints Limit</Label>
                    <Input
                      id="strict-limit"
                      type="text"
                      placeholder="10/minute"
                      value={rateLimit.strictLimit}
                      onChange={(e) => updateRateLimit('strictLimit', e.target.value)}
                      className="font-mono text-xs"
                    />
                    <p className="text-xs text-muted-foreground">
                      Applied to sensitive endpoints (auth, admin, etc.)
                    </p>
                  </div>

                  <div className="p-3 rounded-md border space-y-2">
                    <p className="text-sm font-medium">Current Limits by Endpoint</p>
                    <div className="space-y-1 text-xs font-mono">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">OSINT Free Tools:</span>
                        <span>20/minute</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Jina MCP:</span>
                        <span>20/minute</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">LLM Endpoints:</span>
                        <span>30/minute</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">General API:</span>
                        <span>{rateLimit.defaultLimit}</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </TabsContent>

          {/* Notifications Tab */}
          <TabsContent value="notifications" className="space-y-4 mt-4">
            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 rounded-md border">
                <div>
                  <Label className="text-sm">Success Notifications</Label>
                  <p className="text-xs text-muted-foreground">Show success toasts</p>
                </div>
                <Switch defaultChecked />
              </div>

              <div className="flex items-center justify-between p-3 rounded-md border">
                <div>
                  <Label className="text-sm">Error Notifications</Label>
                  <p className="text-xs text-muted-foreground">Show error alerts</p>
                </div>
                <Switch defaultChecked />
              </div>

              <div className="flex items-center justify-between p-3 rounded-md border">
                <div>
                  <Label className="text-sm">System Alerts</Label>
                  <p className="text-xs text-muted-foreground">Backend status changes</p>
                </div>
                <Switch defaultChecked />
              </div>

              <div className="flex items-center justify-between p-3 rounded-md border">
                <div>
                  <Label className="text-sm">Task Completion</Label>
                  <p className="text-xs text-muted-foreground">Notify when tasks finish</p>
                </div>
                <Switch defaultChecked />
              </div>

              <div className="flex items-center justify-between p-3 rounded-md border">
                <div>
                  <Label className="text-sm">Agent Updates</Label>
                  <p className="text-xs text-muted-foreground">Agent status changes</p>
                </div>
                <Switch defaultChecked />
              </div>

              <div className="p-3 rounded-md bg-blue-500/10 border border-blue-500/20">
                <p className="text-xs text-blue-600 dark:text-blue-400">
                  <Bell className="h-3 w-3 inline mr-1" />
                  Notifications use the Sonner toast library for non-intrusive alerts.
                </p>
              </div>
            </div>
          </TabsContent>
        </Tabs>

        {/* Save Button */}
        <div className="flex justify-end gap-2 pt-4 border-t">
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={saveSettings}>
            Save Settings
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

