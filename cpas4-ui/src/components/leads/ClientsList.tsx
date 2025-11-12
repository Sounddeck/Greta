/**
 * Clients List Component
 * 
 * Displays CRM contacts linked to leads from Lead Generation
 */

import React, { useEffect, useState } from 'react';
import { Users, Building2, MapPin, Mail, Phone, TrendingUp, Badge } from 'lucide-react';
import { api } from '../../lib/api';
import { toast } from 'sonner';

interface Contact {
  contact_id: string;
  first_name: string;
  last_name: string;
  email?: string;
  phone?: string;
  company?: string;
  location?: string;
  status: string;
  custom_fields?: {
    source_lead_id?: string;
  };
  created_at: string;
}

interface Deal {
  deal_id: string;
  title: string;
  value: number;
  stage: string;
}

export default function ClientsList() {
  const [contacts, setContacts] = useState<Contact[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchContacts();
  }, []);

  const fetchContacts = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await api.get('/api/v1/crm/contacts');
      setContacts(response.data.contacts || response.data || []);
    } catch (err: any) {
      if (err.response?.status === 404) {
        setContacts([]);
      } else {
        setError(err.response?.data?.detail || 'Failed to fetch contacts');
        toast.error('Failed to fetch contacts');
      }
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'active':
        return 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300';
      case 'inactive':
        return 'bg-gray-100 dark:bg-gray-900/30 text-gray-800 dark:text-gray-300';
      case 'prospect':
        return 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-300';
      case 'customer':
        return 'bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300';
      case 'archived':
        return 'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-300';
      default:
        return 'bg-gray-100 dark:bg-gray-900/30 text-gray-800 dark:text-gray-300';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
        <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
      </div>
    );
  }

  if (contacts.length === 0) {
    return (
      <div className="text-center py-12 bg-white dark:bg-gray-800 rounded-lg border-2 border-dashed border-gray-300 dark:border-gray-700">
        <Users className="w-12 h-12 text-gray-400 dark:text-gray-600 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
          No clients yet
        </h3>
        <p className="text-gray-600 dark:text-gray-400">
          Classify leads as clients to see them here
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {contacts.map((contact) => (
        <div
          key={contact.contact_id}
          className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:shadow-md transition-shadow"
        >
          <div className="flex items-start justify-between mb-3">
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                {contact.first_name} {contact.last_name}
              </h3>
              {contact.company && (
                <p className="text-sm text-gray-600 dark:text-gray-400 flex items-center gap-2 mt-1">
                  <Building2 className="w-4 h-4" />
                  {contact.company}
                </p>
              )}
            </div>
            <span className={`px-3 py-1 rounded-full text-xs font-medium ${getStatusColor(contact.status)}`}>
              {contact.status}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-3 text-sm">
            {contact.email && (
              <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
                <Mail className="w-4 h-4" />
                <a href={`mailto:${contact.email}`} className="hover:text-blue-600 dark:hover:text-blue-400">
                  {contact.email}
                </a>
              </div>
            )}
            {contact.phone && (
              <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
                <Phone className="w-4 h-4" />
                <a href={`tel:${contact.phone}`} className="hover:text-blue-600 dark:hover:text-blue-400">
                  {contact.phone}
                </a>
              </div>
            )}
            {contact.location && (
              <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
                <MapPin className="w-4 h-4" />
                {contact.location}
              </div>
            )}
          </div>

          {contact.custom_fields?.source_lead_id && (
            <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
              <p className="text-xs text-gray-500 dark:text-gray-400">
                <Badge className="w-3 h-3 inline mr-1" />
                Source Lead: {contact.custom_fields.source_lead_id}
              </p>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

