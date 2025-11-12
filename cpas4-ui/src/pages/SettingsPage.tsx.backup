import React, { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Separator } from '@/components/ui/separator';
import { toast } from 'sonner';
import {
  Key,
  Shield,
  Server,
  Bell,
  User,
  Database,
  Zap,
  Lock,
  Eye,
  EyeOff,
  Save,
  RefreshCw,
  AlertCircle,
  CheckCircle2
} from 'lucide-react';

interface APIKey {
  name: string;
  key: string;
  description: string;
  required: boolean;
  configured: boolean;
}

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState('api-keys');
  const [showSecrets, setShowSecrets] = useState<Record<string, boolean>>({});
  const [hasChanges, setHasChanges] = useState(false);

  // API Keys state
  const [apiKeys, setApiKeys] = useState<Record<string, string>>({
    openai: '',
    anthropic: '',
    jina: '',
    apollo: '',
    clearbit: '',
    hunter: '',
    google_client_id: '',
    google_client_secret: '',
  });

  // Security settings state
  const [securitySettings, setSecuritySettings] = useState({
    mfa_enabled: false,
    password_min_length: 12,
    password_require_uppercase: true,
    password_require_lowercase: true,
    password_require_numbers: true,
    password_require_special: true,
    session_timeout: 60,
    https_redirect: false,
    input_validation: true,
    security_headers: true,
  });

  // OSINT settings state
  const [osintSettings, setOsintSettings] = useState({
    enable_dns_lookup: true,
    enable_whois: true,
    enable_web_scraping: true,
    enable_ssl_check: true,
    use_apollo: false,
    use_clearbit: false,
    use_hunter: false,
  });

  // Rate limiting settings
  const [rateLimitSettings, setRateLimitSettings] = useState({
    max_requests_per_minute: 60,
    max_connections_per_ip: 100,
    enable_rate_limiting: true,
  });

  // Notification settings
  const [notificationSettings, setNotificationSettings] = useState({
    email_notifications: true,
    security_alerts: true,
    system_updates: true,
    agent_completion: true,
  });

  const toggleSecret = (key: string) => {
    setShowSecrets(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const handleAPIKeyChange = (key: string, value: string) => {
    setApiKeys(prev => ({ ...prev, [key]: value }));
    setHasChanges(true);
  };

  const handleSaveSettings = async () => {
    try {
      // TODO: Implement actual API call to save settings
      toast.success('Settings saved successfully');
      setHasChanges(false);
    } catch (error) {
      toast.error('Failed to save settings');
    }
  };

  const handleResetSettings = () => {
    // TODO: Implement reset logic
    toast.info('Settings reset to defaults');
    setHasChanges(false);
  };

  const renderAPIKeyInput = (
    key: string,
    label: string,
    description: string,
    required: boolean = false
  ) => (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label htmlFor={key} className="flex items-center gap-2">
          {label}
          {required && <span className="text-red-500">*</span>}
        </Label>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => toggleSecret(key)}
          className="h-8 w-8 p-0"
        >
          {showSecrets[key] ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
        </Button>
      </div>
      <Input
        id={key}
        type={showSecrets[key] ? 'text' : 'password'}
        value={apiKeys[key]}
        onChange={(e) => handleAPIKeyChange(key, e.target.value)}
        placeholder={`Enter ${label}`}
        className="font-mono text-sm"
      />
      <p className="text-sm text-muted-foreground">{description}</p>
    </div>
  );

  return (
    <div className="container mx-auto p-6 max-w-6xl">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold">Settings</h1>
          <p className="text-muted-foreground mt-1">
            Configure API keys, security, and system preferences
          </p>
        </div>
        <div className="flex gap-2">
          {hasChanges && (
            <Button variant="outline" onClick={handleResetSettings}>
              <RefreshCw className="h-4 w-4 mr-2" />
              Reset
            </Button>
          )}
          <Button onClick={handleSaveSettings} disabled={!hasChanges}>
            <Save className="h-4 w-4 mr-2" />
            Save Changes
          </Button>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="api-keys" className="flex items-center gap-2">
            <Key className="h-4 w-4" />
            API Keys
          </TabsTrigger>
          <TabsTrigger value="security" className="flex items-center gap-2">
            <Shield className="h-4 w-4" />
            Security
          </TabsTrigger>
          <TabsTrigger value="osint" className="flex items-center gap-2">
            <Database className="h-4 w-4" />
            OSINT
          </TabsTrigger>
          <TabsTrigger value="mcp" className="flex items-center gap-2">
            <Server className="h-4 w-4" />
            MCP Servers
          </TabsTrigger>
          <TabsTrigger value="rate-limit" className="flex items-center gap-2">
            <Zap className="h-4 w-4" />
            Rate Limiting
          </TabsTrigger>
          <TabsTrigger value="notifications" className="flex items-center gap-2">
            <Bell className="h-4 w-4" />
            Notifications
          </TabsTrigger>
        </TabsList>

        {/* API Keys Tab */}
        <TabsContent value="api-keys" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Key className="h-5 w-5" />
                LLM Provider API Keys
              </CardTitle>
              <CardDescription>
                Configure API keys for OpenAI, Anthropic, and other LLM providers
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {renderAPIKeyInput(
                'openai',
                'OpenAI API Key',
                'Required for GPT-4, GPT-3.5, and other OpenAI models',
                true
              )}
              <Separator />
              {renderAPIKeyInput(
                'anthropic',
                'Anthropic API Key',
                'Required for Claude models (Claude 3.5 Sonnet, etc.)',
                true
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Server className="h-5 w-5" />
                MCP & Search API Keys
              </CardTitle>
              <CardDescription>
                Configure API keys for MCP servers and search services
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {renderAPIKeyInput(
                'jina',
                'Jina AI API Key',
                'System-wide MCP server for search and embeddings'
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Database className="h-5 w-5" />
                OSINT & Lead Generation API Keys
              </CardTitle>
              <CardDescription>
                Configure API keys for OSINT tools and lead generation services
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {renderAPIKeyInput(
                'apollo',
                'Apollo.io API Key',
                'Company enrichment and contact finding'
              )}
              <Separator />
              {renderAPIKeyInput(
                'clearbit',
                'Clearbit API Key',
                'Company and person enrichment ($99-999/month)'
              )}
              <Separator />
              {renderAPIKeyInput(
                'hunter',
                'Hunter.io API Key',
                'Email finding and verification'
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Lock className="h-5 w-5" />
                Google Workspace OAuth
              </CardTitle>
              <CardDescription>
                Configure Google OAuth for Gmail, Calendar, and Drive integration
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {renderAPIKeyInput(
                'google_client_id',
                'Google Client ID',
                'OAuth 2.0 Client ID from Google Cloud Console'
              )}
              <Separator />
              {renderAPIKeyInput(
                'google_client_secret',
                'Google Client Secret',
                'OAuth 2.0 Client Secret from Google Cloud Console'
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Security Tab */}
        <TabsContent value="security" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="h-5 w-5" />
                Security Settings
              </CardTitle>
              <CardDescription>
                Configure password policy, MFA, and security features
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Multi-Factor Authentication (MFA)</Label>
                  <p className="text-sm text-muted-foreground">
                    Require TOTP for all user logins
                  </p>
                </div>
                <Switch
                  checked={securitySettings.mfa_enabled}
                  onCheckedChange={(checked) => {
                    setSecuritySettings(prev => ({ ...prev, mfa_enabled: checked }));
                    setHasChanges(true);
                  }}
                />
              </div>

              <Separator />

              <div className="space-y-2">
                <Label>Minimum Password Length</Label>
                <Input
                  type="number"
                  min="8"
                  max="32"
                  value={securitySettings.password_min_length}
                  onChange={(e) => {
                    setSecuritySettings(prev => ({
                      ...prev,
                      password_min_length: parseInt(e.target.value)
                    }));
                    setHasChanges(true);
                  }}
                />
                <p className="text-sm text-muted-foreground">
                  Current: {securitySettings.password_min_length} characters (recommended: 12+)
                </p>
              </div>

              <Separator />

              <div className="space-y-4">
                <Label>Password Requirements</Label>
                {[
                  { key: 'password_require_uppercase', label: 'Require uppercase letters' },
                  { key: 'password_require_lowercase', label: 'Require lowercase letters' },
                  { key: 'password_require_numbers', label: 'Require numbers' },
                  { key: 'password_require_special', label: 'Require special characters' },
                ].map(({ key, label }) => (
                  <div key={key} className="flex items-center justify-between">
                    <Label className="font-normal">{label}</Label>
                    <Switch
                      checked={securitySettings[key as keyof typeof securitySettings] as boolean}
                      onCheckedChange={(checked) => {
                        setSecuritySettings(prev => ({ ...prev, [key]: checked }));
                        setHasChanges(true);
                      }}
                    />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Other tabs will be implemented similarly */}
        <TabsContent value="osint">
          <Card>
            <CardHeader>
              <CardTitle>OSINT Configuration</CardTitle>
              <CardDescription>Configure OSINT tools and data sources</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground">OSINT settings coming soon...</p>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="mcp">
          <Card>
            <CardHeader>
              <CardTitle>MCP Server Configuration</CardTitle>
              <CardDescription>Manage system-wide and per-agent MCP servers</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground">MCP settings coming soon...</p>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="rate-limit">
          <Card>
            <CardHeader>
              <CardTitle>Rate Limiting</CardTitle>
              <CardDescription>Configure API rate limits and DDoS protection</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground">Rate limiting settings coming soon...</p>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="notifications">
          <Card>
            <CardHeader>
              <CardTitle>Notification Preferences</CardTitle>
              <CardDescription>Configure email and system notifications</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground">Notification settings coming soon...</p>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

