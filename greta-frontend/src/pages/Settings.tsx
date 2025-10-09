import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Settings as SettingsIcon,
  User,
  Bell,
  Shield,
  Palette,
  Database,
  Mic,
  Globe,
  Save,
  RefreshCw
} from 'lucide-react';

interface SettingSection {
  id: string;
  title: string;
  icon: React.ElementType;
  description: string;
}

const settingSections: SettingSection[] = [
  {
    id: 'profile',
    title: 'Profile',
    icon: User,
    description: 'Manage your personal information and preferences'
  },
  {
    id: 'notifications',
    title: 'Notifications',
    icon: Bell,
    description: 'Configure notification preferences'
  },
  {
    id: 'security',
    title: 'Security',
    icon: Shield,
    description: 'Security and privacy settings'
  },
  {
    id: 'appearance',
    title: 'Appearance',
    icon: Palette,
    description: 'Customize the look and feel'
  },
  {
    id: 'voice',
    title: 'Voice',
    icon: Mic,
    description: 'Voice recognition and synthesis settings'
  },
  {
    id: 'data',
    title: 'Data',
    icon: Database,
    description: 'Data management and storage'
  },
  {
    id: 'language',
    title: 'Language',
    icon: Globe,
    description: 'Language and localization settings'
  }
];

const SettingCard: React.FC<{ section: SettingSection; isActive: boolean; onClick: () => void }> = ({
  section,
  isActive,
  onClick
}) => {
  const Icon = section.icon;
  
  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      onClick={onClick}
      className={`p-4 rounded-lg cursor-pointer transition-all ${
        isActive
          ? 'bg-gradient-to-r from-purple-600/20 to-blue-600/20 border border-purple-500/30'
          : 'glass hover:bg-blue-500/15'
      }`}
    >
      <div className="flex items-center space-x-3">
        <div className={`p-2 rounded-lg ${
          isActive ? 'bg-purple-600' : 'bg-blue-500/15'
        }`}>
          <Icon className="w-5 h-5 text-white" />
        </div>
        <div>
          <h3 className="font-medium text-white">{section.title}</h3>
          <p className="text-sm text-gray-400">{section.description}</p>
        </div>
      </div>
    </motion.div>
  );
};

const ProfileSettings: React.FC = () => (
  <div className="space-y-6">
    <h2 className="text-xl font-bold text-white">Profile Settings</h2>
    
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Display Name
        </label>
        <input
          type="text"
          defaultValue="GRETA User"
          className="input-field w-full"
        />
      </div>
      
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Email
        </label>
        <input
          type="email"
          defaultValue="user@example.com"
          className="input-field w-full"
        />
      </div>
      
      <div className="md:col-span-2">
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Bio
        </label>
        <textarea
          rows={3}
          defaultValue="AI enthusiast working with GRETA PAI system"
          className="input-field w-full"
        />
      </div>
    </div>
  </div>
);

const NotificationSettings: React.FC = () => (
  <div className="space-y-6">
    <h2 className="text-xl font-bold text-white">Notification Settings</h2>
    
    <div className="space-y-4">
      {[
        { id: 'agent-updates', label: 'Agent Status Updates', description: 'Get notified when agents start, stop, or encounter errors' },
        { id: 'workflow-completion', label: 'Workflow Completion', description: 'Notifications when workflows complete or fail' },
        { id: 'system-alerts', label: 'System Alerts', description: 'Important system notifications and alerts' },
        { id: 'chat-messages', label: 'Chat Messages', description: 'Notifications for new chat messages' }
      ].map((setting) => (
        <div key={setting.id} className="flex items-center justify-between p-4 glass rounded-lg">
          <div>
            <h3 className="font-medium text-white">{setting.label}</h3>
            <p className="text-sm text-gray-400">{setting.description}</p>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input type="checkbox" className="sr-only peer" defaultChecked />
            <div className="w-11 h-6 bg-blue-700 hover:bg-blue-800 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-600"></div>
          </label>
        </div>
      ))}
    </div>
  </div>
);

const AppearanceSettings: React.FC = () => (
  <div className="space-y-6">
    <h2 className="text-xl font-bold text-white">Appearance Settings</h2>
    
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Theme
        </label>
        <select className="input-field w-full">
          <option>Dark (Default)</option>
          <option>Light</option>
          <option>Auto</option>
        </select>
      </div>
      
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Accent Color
        </label>
        <div className="flex space-x-3">
          {['purple', 'blue', 'green', 'red', 'yellow'].map((color) => (
            <button
              key={color}
              className={`w-8 h-8 rounded-full bg-${color}-600 border-2 border-transparent hover:border-white transition-colors`}
            />
          ))}
        </div>
      </div>
      
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Animation Level
        </label>
        <select className="input-field w-full">
          <option>Full Animations</option>
          <option>Reduced Animations</option>
          <option>No Animations</option>
        </select>
      </div>
    </div>
  </div>
);

export const Settings: React.FC = () => {
  const [activeSection, setActiveSection] = useState('profile');

  const renderSettingContent = () => {
    switch (activeSection) {
      case 'profile':
        return <ProfileSettings />;
      case 'notifications':
        return <NotificationSettings />;
      case 'appearance':
        return <AppearanceSettings />;
      default:
        return (
          <div className="text-center py-12">
            <SettingsIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-400">Select a setting category to configure</p>
          </div>
        );
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">⚙️ Settings</h1>
          <p className="text-gray-400">Customize your GRETA experience</p>
        </div>
        
        <div className="flex items-center space-x-3">
          <button className="btn-secondary flex items-center space-x-2">
            <RefreshCw className="w-4 h-4" />
            <span>Reset</span>
          </button>
          <button className="btn-primary flex items-center space-x-2">
            <Save className="w-4 h-4" />
            <span>Save Changes</span>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Settings Navigation */}
        <div className="lg:col-span-1">
          <div className="space-y-2">
            {settingSections.map((section) => (
              <SettingCard
                key={section.id}
                section={section}
                isActive={activeSection === section.id}
                onClick={() => setActiveSection(section.id)}
              />
            ))}
          </div>
        </div>

        {/* Settings Content */}
        <div className="lg:col-span-3">
          <motion.div
            key={activeSection}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3 }}
            className="card"
          >
            {renderSettingContent()}
          </motion.div>
        </div>
      </div>
    </div>
  );
};
