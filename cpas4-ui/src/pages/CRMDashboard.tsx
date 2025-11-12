import { useEffect, useState, useRef } from 'react';
import { useCRMStore } from '../stores/crmStore';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Badge } from '../components/ui/badge';
import {
  Users,
  Building2,
  DollarSign,
  TrendingUp,
  Plus,
  Search,
  Mail,
  Phone,
  Calendar,
  Filter,
  Download,
  RefreshCw,
  Upload,
} from 'lucide-react';
import { toast } from 'sonner';
import CreateContactModal from '../components/crm/CreateContactModal';
import { CreateCompanyModal } from '../components/crm/CreateCompanyModal';
import { CreateDealModal } from '../components/crm/CreateDealModal';
import { exportContactsToCSV, exportCompaniesToCSV, exportDealsToCSV } from '../utils/exportUtils';

export default function CRMDashboard() {
  const {
    contacts,
    companies,
    deals,
    stats,
    isLoading,
    fetchContacts,
    fetchCompanies,
    fetchDeals,
    fetchStats,
    importContactsCSV,
    importCompaniesCSV,
  } = useCRMStore();

  const [searchQuery, setSearchQuery] = useState('');
  const [activeTab, setActiveTab] = useState<'contacts' | 'companies' | 'deals'>('contacts');
  const [showCreateContactModal, setShowCreateContactModal] = useState(false);
  const [showCreateCompanyModal, setShowCreateCompanyModal] = useState(false);
  const [showCreateDealModal, setShowCreateDealModal] = useState(false);
  const contactsFileInputRef = useRef<HTMLInputElement>(null);
  const companiesFileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Load data on mount
    fetchContacts();
    fetchCompanies();
    fetchDeals();
    fetchStats();
  }, []);

  const handleRefresh = () => {
    fetchContacts();
    fetchCompanies();
    fetchDeals();
    fetchStats();
  };

  const handleContactsCSVImport = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.name.endsWith('.csv')) {
      toast.error('Please select a CSV file');
      return;
    }

    await importContactsCSV(file);

    // Reset file input
    if (contactsFileInputRef.current) {
      contactsFileInputRef.current.value = '';
    }
  };

  const handleCompaniesCSVImport = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.name.endsWith('.csv')) {
      toast.error('Please select a CSV file');
      return;
    }

    await importCompaniesCSV(file);

    // Reset file input
    if (companiesFileInputRef.current) {
      companiesFileInputRef.current.value = '';
    }
  };

  const handleExport = () => {
    try {
      if (activeTab === 'contacts') {
        exportContactsToCSV(contacts);
        toast.success('Contacts exported successfully');
      } else if (activeTab === 'companies') {
        exportCompaniesToCSV(companies);
        toast.success('Companies exported successfully');
      } else if (activeTab === 'deals') {
        exportDealsToCSV(deals);
        toast.success('Deals exported successfully');
      }
    } catch (error) {
      toast.error('Failed to export data');
      console.error(error);
    }
  };

  const handleAddNew = () => {
    if (activeTab === 'contacts') {
      setShowCreateContactModal(true);
    } else if (activeTab === 'companies') {
      setShowCreateCompanyModal(true);
    } else if (activeTab === 'deals') {
      setShowCreateDealModal(true);
    }
  };

  const getStatusColor = (status: string) => {
    const colors: Record<string, string> = {
      lead: 'bg-blue-500',
      prospect: 'bg-yellow-500',
      customer: 'bg-green-500',
      inactive: 'bg-gray-500',
      lost: 'bg-red-500',
    };
    return colors[status] || 'bg-gray-500';
  };

  const getDealStageColor = (stage: string) => {
    const colors: Record<string, string> = {
      lead: 'bg-blue-500',
      qualified: 'bg-cyan-500',
      proposal: 'bg-yellow-500',
      negotiation: 'bg-orange-500',
      closed_won: 'bg-green-500',
      closed_lost: 'bg-red-500',
    };
    return colors[stage] || 'bg-gray-500';
  };

  const filteredContacts = contacts.filter(
    (contact) =>
      contact.first_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      contact.last_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      contact.email.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const filteredCompanies = companies.filter((company) =>
    company.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const filteredDeals = deals.filter((deal) =>
    deal.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="h-full overflow-auto bg-background p-6">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-3xl font-bold text-foreground">CRM Dashboard</h1>
            <p className="text-muted-foreground mt-1">
              Manage your customer relationships and sales pipeline
            </p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={handleRefresh}>
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh
            </Button>
            <Button variant="outline" size="sm" onClick={handleExport}>
              <Download className="h-4 w-4 mr-2" />
              Export
            </Button>
            <Button size="sm" onClick={handleAddNew}>
              <Plus className="h-4 w-4 mr-2" />
              Add New
            </Button>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <Card className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Total Contacts</p>
                <p className="text-2xl font-bold">{stats?.total_contacts || 0}</p>
              </div>
              <Users className="h-8 w-8 text-blue-500" />
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Companies</p>
                <p className="text-2xl font-bold">{stats?.total_companies || 0}</p>
              </div>
              <Building2 className="h-8 w-8 text-purple-500" />
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Active Deals</p>
                <p className="text-2xl font-bold">{stats?.total_deals || 0}</p>
              </div>
              <TrendingUp className="h-8 w-8 text-green-500" />
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Deal Value</p>
                <p className="text-2xl font-bold">
                  ${(stats?.total_deal_value || 0).toLocaleString()}
                </p>
              </div>
              <DollarSign className="h-8 w-8 text-yellow-500" />
            </div>
          </Card>
        </div>

        {/* Search and Filter */}
        <div className="flex gap-4 mb-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>
          <Button variant="outline">
            <Filter className="h-4 w-4 mr-2" />
            Filters
          </Button>

          {/* CSV Import Buttons */}
          {activeTab === 'contacts' && (
            <>
              <input
                ref={contactsFileInputRef}
                type="file"
                accept=".csv"
                onChange={handleContactsCSVImport}
                className="hidden"
              />
              <Button
                variant="outline"
                onClick={() => contactsFileInputRef.current?.click()}
                disabled={isLoading}
              >
                <Upload className="h-4 w-4 mr-2" />
                Import CSV
              </Button>
            </>
          )}

          {activeTab === 'companies' && (
            <>
              <input
                ref={companiesFileInputRef}
                type="file"
                accept=".csv"
                onChange={handleCompaniesCSVImport}
                className="hidden"
              />
              <Button
                variant="outline"
                onClick={() => companiesFileInputRef.current?.click()}
                disabled={isLoading}
              >
                <Upload className="h-4 w-4 mr-2" />
                Import CSV
              </Button>
            </>
          )}
        </div>

        {/* Tabs */}
        <div className="flex gap-2 border-b border-border">
          <button
            onClick={() => setActiveTab('contacts')}
            className={`px-4 py-2 font-medium transition-colors ${
              activeTab === 'contacts'
                ? 'text-primary border-b-2 border-primary'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            Contacts ({contacts.length})
          </button>
          <button
            onClick={() => setActiveTab('companies')}
            className={`px-4 py-2 font-medium transition-colors ${
              activeTab === 'companies'
                ? 'text-primary border-b-2 border-primary'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            Companies ({companies.length})
          </button>
          <button
            onClick={() => setActiveTab('deals')}
            className={`px-4 py-2 font-medium transition-colors ${
              activeTab === 'deals'
                ? 'text-primary border-b-2 border-primary'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            Deals ({deals.length})
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="space-y-4">
        {isLoading ? (
          <div className="text-center py-12">
            <RefreshCw className="h-8 w-8 animate-spin mx-auto text-muted-foreground" />
            <p className="text-muted-foreground mt-2">Loading...</p>
          </div>
        ) : (
          <>
            {/* Contacts Tab */}
            {activeTab === 'contacts' && (
              <div className="grid grid-cols-1 gap-4">
                {filteredContacts.length === 0 ? (
                  <Card className="p-8 text-center">
                    <Users className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                    <p className="text-muted-foreground">No contacts found</p>
                    <Button className="mt-4" size="sm">
                      <Plus className="h-4 w-4 mr-2" />
                      Add First Contact
                    </Button>
                  </Card>
                ) : (
                  filteredContacts.map((contact) => (
                    <Card key={contact.contact_id} className="p-4 hover:shadow-md transition-shadow">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-2">
                            <h3 className="text-lg font-semibold">
                              {contact.first_name} {contact.last_name}
                            </h3>
                            <Badge className={getStatusColor(contact.status)}>
                              {contact.status}
                            </Badge>
                            {contact.lead_score > 70 && (
                              <Badge variant="outline" className="bg-yellow-500/10 text-yellow-700">
                                High Score: {contact.lead_score}
                              </Badge>
                            )}
                          </div>
                          <div className="flex flex-wrap gap-4 text-sm text-muted-foreground">
                            {contact.title && <span>{contact.title}</span>}
                            <span className="flex items-center gap-1">
                              <Mail className="h-3 w-3" />
                              {contact.email}
                            </span>
                            {contact.phone && (
                              <span className="flex items-center gap-1">
                                <Phone className="h-3 w-3" />
                                {contact.phone}
                              </span>
                            )}
                          </div>
                          {contact.tags.length > 0 && (
                            <div className="flex gap-2 mt-2">
                              {contact.tags.map((tag) => (
                                <Badge key={tag} variant="secondary" className="text-xs">
                                  {tag}
                                </Badge>
                              ))}
                            </div>
                          )}
                        </div>
                        <Button variant="outline" size="sm">
                          View Details
                        </Button>
                      </div>
                    </Card>
                  ))
                )}
              </div>
            )}

            {/* Companies Tab */}
            {activeTab === 'companies' && (
              <div className="grid grid-cols-1 gap-4">
                {filteredCompanies.length === 0 ? (
                  <Card className="p-8 text-center">
                    <Building2 className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                    <p className="text-muted-foreground">No companies found</p>
                    <Button className="mt-4" size="sm">
                      <Plus className="h-4 w-4 mr-2" />
                      Add First Company
                    </Button>
                  </Card>
                ) : (
                  filteredCompanies.map((company) => (
                    <Card key={company.company_id} className="p-4 hover:shadow-md transition-shadow">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <h3 className="text-lg font-semibold mb-2">{company.name}</h3>
                          <div className="flex flex-wrap gap-4 text-sm text-muted-foreground">
                            {company.industry && <span>{company.industry}</span>}
                            {company.size && <span>{company.size} employees</span>}
                            {company.city && company.country && (
                              <span>
                                {company.city}, {company.country}
                              </span>
                            )}
                          </div>
                          {company.tags.length > 0 && (
                            <div className="flex gap-2 mt-2">
                              {company.tags.map((tag) => (
                                <Badge key={tag} variant="secondary" className="text-xs">
                                  {tag}
                                </Badge>
                              ))}
                            </div>
                          )}
                        </div>
                        <Button variant="outline" size="sm">
                          View Details
                        </Button>
                      </div>
                    </Card>
                  ))
                )}
              </div>
            )}

            {/* Deals Tab */}
            {activeTab === 'deals' && (
              <div className="grid grid-cols-1 gap-4">
                {filteredDeals.length === 0 ? (
                  <Card className="p-8 text-center">
                    <DollarSign className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                    <p className="text-muted-foreground">No deals found</p>
                    <Button className="mt-4" size="sm">
                      <Plus className="h-4 w-4 mr-2" />
                      Create First Deal
                    </Button>
                  </Card>
                ) : (
                  filteredDeals.map((deal) => (
                    <Card key={deal.deal_id} className="p-4 hover:shadow-md transition-shadow">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-2">
                            <h3 className="text-lg font-semibold">{deal.title}</h3>
                            <Badge className={getDealStageColor(deal.stage)}>
                              {deal.stage.replace('_', ' ')}
                            </Badge>
                          </div>
                          <div className="flex flex-wrap gap-4 text-sm text-muted-foreground">
                            <span className="font-semibold text-foreground">
                              {deal.currency} {deal.value.toLocaleString()}
                            </span>
                            <span>Probability: {deal.probability}%</span>
                            {deal.expected_close_date && (
                              <span className="flex items-center gap-1">
                                <Calendar className="h-3 w-3" />
                                Close: {new Date(deal.expected_close_date).toLocaleDateString()}
                              </span>
                            )}
                          </div>
                        </div>
                        <Button variant="outline" size="sm">
                          View Details
                        </Button>
                      </div>
                    </Card>
                  ))
                )}
              </div>
            )}
          </>
        )}
      </div>

      {/* Modals */}
      {showCreateContactModal && (
        <CreateContactModal onClose={() => setShowCreateContactModal(false)} />
      )}
      {showCreateCompanyModal && (
        <CreateCompanyModal onClose={() => setShowCreateCompanyModal(false)} />
      )}
      {showCreateDealModal && (
        <CreateDealModal onClose={() => setShowCreateDealModal(false)} />
      )}
    </div>
  );
}

