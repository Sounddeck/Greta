import { useEffect, useState, useRef } from 'react';
import { useATSStore } from '../stores/atsStore';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Badge } from '../components/ui/badge';
import {
  Briefcase,
  Users,
  FileText,
  Calendar,
  Plus,
  Search,
  Filter,
  Download,
  RefreshCw,
  MapPin,
  DollarSign,
  Clock,
  Upload,
} from 'lucide-react';
import { toast } from 'sonner';
import CreateJobModal from '../components/ats/CreateJobModal';
import { CreateCandidateModal } from '../components/ats/CreateCandidateModal';
import { CreateApplicationModal } from '../components/ats/CreateApplicationModal';
import { CreateInterviewModal } from '../components/ats/CreateInterviewModal';
import { exportJobsToCSV, exportCandidatesToCSV, exportApplicationsToCSV, exportInterviewsToCSV } from '../utils/exportUtils';

export default function ATSDashboard() {
  const {
    jobs,
    candidates,
    applications,
    interviews,
    stats,
    isLoading,
    fetchJobs,
    fetchCandidates,
    fetchApplications,
    fetchInterviews,
    fetchStats,
    importCandidatesCSV,
  } = useATSStore();

  const [searchQuery, setSearchQuery] = useState('');
  const [activeTab, setActiveTab] = useState<'jobs' | 'candidates' | 'applications' | 'interviews'>('jobs');
  const [showCreateJobModal, setShowCreateJobModal] = useState(false);
  const [showCreateCandidateModal, setShowCreateCandidateModal] = useState(false);
  const [showCreateApplicationModal, setShowCreateApplicationModal] = useState(false);
  const [showCreateInterviewModal, setShowCreateInterviewModal] = useState(false);
  const candidatesFileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Load data on mount
    fetchJobs();
    fetchCandidates();
    fetchApplications();
    fetchInterviews();
    fetchStats();
  }, []);

  const handleRefresh = () => {
    fetchJobs();
    fetchCandidates();
    fetchApplications();
    fetchInterviews();
    fetchStats();
  };

  const handleCandidatesCSVImport = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.name.endsWith('.csv')) {
      toast.error('Please select a CSV file');
      return;
    }

    await importCandidatesCSV(file);

    // Reset file input
    if (candidatesFileInputRef.current) {
      candidatesFileInputRef.current.value = '';
    }
  };

  const handleExport = () => {
    try {
      if (activeTab === 'jobs') {
        exportJobsToCSV(jobs);
        toast.success('Jobs exported successfully');
      } else if (activeTab === 'candidates') {
        exportCandidatesToCSV(candidates);
        toast.success('Candidates exported successfully');
      } else if (activeTab === 'applications') {
        exportApplicationsToCSV(applications);
        toast.success('Applications exported successfully');
      } else if (activeTab === 'interviews') {
        exportInterviewsToCSV(interviews);
        toast.success('Interviews exported successfully');
      }
    } catch (error) {
      toast.error('Failed to export data');
      console.error(error);
    }
  };

  const handleAddNew = () => {
    if (activeTab === 'jobs') {
      setShowCreateJobModal(true);
    } else if (activeTab === 'candidates') {
      setShowCreateCandidateModal(true);
    } else if (activeTab === 'applications') {
      setShowCreateApplicationModal(true);
    } else if (activeTab === 'interviews') {
      setShowCreateInterviewModal(true);
    }
  };

  const getJobStatusColor = (status: string) => {
    const colors: Record<string, string> = {
      draft: 'bg-gray-500',
      open: 'bg-green-500',
      closed: 'bg-red-500',
      on_hold: 'bg-yellow-500',
      filled: 'bg-blue-500',
    };
    return colors[status] || 'bg-gray-500';
  };

  const getApplicationStatusColor = (status: string) => {
    const colors: Record<string, string> = {
      applied: 'bg-blue-500',
      screening: 'bg-cyan-500',
      phone_screen: 'bg-purple-500',
      interview: 'bg-yellow-500',
      technical_test: 'bg-orange-500',
      final_interview: 'bg-pink-500',
      offer: 'bg-green-500',
      hired: 'bg-emerald-500',
      rejected: 'bg-red-500',
      withdrawn: 'bg-gray-500',
    };
    return colors[status] || 'bg-gray-500';
  };

  const filteredJobs = jobs.filter((job) =>
    job.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const filteredCandidates = candidates.filter(
    (candidate) =>
      candidate.first_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      candidate.last_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      candidate.email.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const filteredApplications = applications.filter((app) => {
    const job = jobs.find((j) => j.job_id === app.job_id);
    const candidate = candidates.find((c) => c.candidate_id === app.candidate_id);
    return (
      job?.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      candidate?.first_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      candidate?.last_name.toLowerCase().includes(searchQuery.toLowerCase())
    );
  });

  return (
    <div className="h-full overflow-auto bg-background p-6">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-3xl font-bold text-foreground">ATS Dashboard</h1>
            <p className="text-muted-foreground mt-1">
              Manage job postings, candidates, and hiring pipeline
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
                <p className="text-sm text-muted-foreground">Total Jobs</p>
                <p className="text-2xl font-bold">{stats?.total_jobs || 0}</p>
                <p className="text-xs text-green-600 mt-1">
                  {stats?.open_jobs || 0} open positions
                </p>
              </div>
              <Briefcase className="h-8 w-8 text-blue-500" />
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Candidates</p>
                <p className="text-2xl font-bold">{stats?.total_candidates || 0}</p>
              </div>
              <Users className="h-8 w-8 text-purple-500" />
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Applications</p>
                <p className="text-2xl font-bold">{stats?.total_applications || 0}</p>
                <p className="text-xs text-blue-600 mt-1">
                  {stats?.recent_applications || 0} this week
                </p>
              </div>
              <FileText className="h-8 w-8 text-green-500" />
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Interviews</p>
                <p className="text-2xl font-bold">{stats?.interviews_scheduled || 0}</p>
                <p className="text-xs text-orange-600 mt-1">scheduled</p>
              </div>
              <Calendar className="h-8 w-8 text-yellow-500" />
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

          {/* CSV Import Button for Candidates */}
          {activeTab === 'candidates' && (
            <>
              <input
                ref={candidatesFileInputRef}
                type="file"
                accept=".csv"
                onChange={handleCandidatesCSVImport}
                className="hidden"
              />
              <Button
                variant="outline"
                onClick={() => candidatesFileInputRef.current?.click()}
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
            onClick={() => setActiveTab('jobs')}
            className={`px-4 py-2 font-medium transition-colors ${
              activeTab === 'jobs'
                ? 'text-primary border-b-2 border-primary'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            Jobs ({jobs.length})
          </button>
          <button
            onClick={() => setActiveTab('candidates')}
            className={`px-4 py-2 font-medium transition-colors ${
              activeTab === 'candidates'
                ? 'text-primary border-b-2 border-primary'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            Candidates ({candidates.length})
          </button>
          <button
            onClick={() => setActiveTab('applications')}
            className={`px-4 py-2 font-medium transition-colors ${
              activeTab === 'applications'
                ? 'text-primary border-b-2 border-primary'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            Applications ({applications.length})
          </button>
          <button
            onClick={() => setActiveTab('interviews')}
            className={`px-4 py-2 font-medium transition-colors ${
              activeTab === 'interviews'
                ? 'text-primary border-b-2 border-primary'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            Interviews ({interviews.length})
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
            {/* Jobs Tab */}
            {activeTab === 'jobs' && (
              <div className="grid grid-cols-1 gap-4">
                {filteredJobs.length === 0 ? (
                  <Card className="p-8 text-center">
                    <Briefcase className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                    <p className="text-muted-foreground">No jobs found</p>
                    <Button className="mt-4" size="sm">
                      <Plus className="h-4 w-4 mr-2" />
                      Post First Job
                    </Button>
                  </Card>
                ) : (
                  filteredJobs.map((job) => (
                    <Card key={job.job_id} className="p-4 hover:shadow-md transition-shadow">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-2">
                            <h3 className="text-lg font-semibold">{job.title}</h3>
                            <Badge className={getJobStatusColor(job.status)}>
                              {job.status.replace('_', ' ')}
                            </Badge>
                            {job.remote_allowed && (
                              <Badge variant="outline" className="bg-blue-500/10 text-blue-700">
                                Remote
                              </Badge>
                            )}
                          </div>
                          <div className="flex flex-wrap gap-4 text-sm text-muted-foreground mb-2">
                            <span className="flex items-center gap-1">
                              <MapPin className="h-3 w-3" />
                              {job.location}
                            </span>
                            <span className="flex items-center gap-1">
                              <Briefcase className="h-3 w-3" />
                              {job.job_type.replace('_', ' ')}
                            </span>
                            <span className="flex items-center gap-1">
                              <Clock className="h-3 w-3" />
                              {job.experience_level}
                            </span>
                            {job.salary_min && job.salary_max && (
                              <span className="flex items-center gap-1">
                                <DollarSign className="h-3 w-3" />
                                {job.salary_currency} {job.salary_min.toLocaleString()} -{' '}
                                {job.salary_max.toLocaleString()}
                              </span>
                            )}
                          </div>
                          <div className="flex gap-4 text-xs text-muted-foreground">
                            <span>{job.application_count} applications</span>
                            <span>{job.view_count} views</span>
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

            {/* Candidates Tab */}
            {activeTab === 'candidates' && (
              <div className="grid grid-cols-1 gap-4">
                {filteredCandidates.length === 0 ? (
                  <Card className="p-8 text-center">
                    <Users className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                    <p className="text-muted-foreground">No candidates found</p>
                  </Card>
                ) : (
                  filteredCandidates.map((candidate) => (
                    <Card key={candidate.candidate_id} className="p-4 hover:shadow-md transition-shadow">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-2">
                            <h3 className="text-lg font-semibold">
                              {candidate.first_name} {candidate.last_name}
                            </h3>
                            <Badge>{candidate.status}</Badge>
                          </div>
                          <div className="flex flex-wrap gap-4 text-sm text-muted-foreground">
                            {candidate.current_title && <span>{candidate.current_title}</span>}
                            {candidate.current_company && <span>at {candidate.current_company}</span>}
                            {candidate.location && (
                              <span className="flex items-center gap-1">
                                <MapPin className="h-3 w-3" />
                                {candidate.location}
                              </span>
                            )}
                            {candidate.experience_years && (
                              <span>{candidate.experience_years} years experience</span>
                            )}
                          </div>
                          {candidate.skills.length > 0 && (
                            <div className="flex gap-2 mt-2 flex-wrap">
                              {candidate.skills.slice(0, 5).map((skill) => (
                                <Badge key={skill} variant="secondary" className="text-xs">
                                  {skill}
                                </Badge>
                              ))}
                              {candidate.skills.length > 5 && (
                                <Badge variant="secondary" className="text-xs">
                                  +{candidate.skills.length - 5} more
                                </Badge>
                              )}
                            </div>
                          )}
                        </div>
                        <Button variant="outline" size="sm">
                          View Profile
                        </Button>
                      </div>
                    </Card>
                  ))
                )}
              </div>
            )}

            {/* Applications Tab */}
            {activeTab === 'applications' && (
              <div className="grid grid-cols-1 gap-4">
                {filteredApplications.length === 0 ? (
                  <Card className="p-8 text-center">
                    <FileText className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                    <p className="text-muted-foreground">No applications found</p>
                  </Card>
                ) : (
                  filteredApplications.map((application) => {
                    const job = jobs.find((j) => j.job_id === application.job_id);
                    const candidate = candidates.find((c) => c.candidate_id === application.candidate_id);
                    return (
                      <Card key={application.application_id} className="p-4 hover:shadow-md transition-shadow">
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <div className="flex items-center gap-3 mb-2">
                              <h3 className="text-lg font-semibold">
                                {candidate?.first_name} {candidate?.last_name}
                              </h3>
                              <Badge className={getApplicationStatusColor(application.status)}>
                                {application.status.replace('_', ' ')}
                              </Badge>
                              {application.match_score > 80 && (
                                <Badge variant="outline" className="bg-green-500/10 text-green-700">
                                  {application.match_score}% match
                                </Badge>
                              )}
                            </div>
                            <p className="text-sm text-muted-foreground mb-2">
                              Applied for: <span className="font-medium text-foreground">{job?.title}</span>
                            </p>
                            <p className="text-xs text-muted-foreground">
                              Applied {new Date(application.applied_at).toLocaleDateString()}
                            </p>
                          </div>
                          <Button variant="outline" size="sm">
                            Review
                          </Button>
                        </div>
                      </Card>
                    );
                  })
                )}
              </div>
            )}

            {/* Interviews Tab */}
            {activeTab === 'interviews' && (
              <div className="grid grid-cols-1 gap-4">
                {interviews.length === 0 ? (
                  <Card className="p-8 text-center">
                    <Calendar className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                    <p className="text-muted-foreground">No interviews scheduled</p>
                  </Card>
                ) : (
                  interviews.map((interview) => {
                    const candidate = candidates.find((c) => c.candidate_id === interview.candidate_id);
                    const job = jobs.find((j) => j.job_id === interview.job_id);
                    return (
                      <Card key={interview.interview_id} className="p-4 hover:shadow-md transition-shadow">
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <div className="flex items-center gap-3 mb-2">
                              <h3 className="text-lg font-semibold">
                                {candidate?.first_name} {candidate?.last_name}
                              </h3>
                              <Badge>{interview.type.replace('_', ' ')}</Badge>
                              <Badge variant="outline">{interview.status}</Badge>
                            </div>
                            <p className="text-sm text-muted-foreground mb-2">
                              Position: <span className="font-medium text-foreground">{job?.title}</span>
                            </p>
                            <div className="flex gap-4 text-xs text-muted-foreground">
                              <span className="flex items-center gap-1">
                                <Calendar className="h-3 w-3" />
                                {new Date(interview.scheduled_at).toLocaleString()}
                              </span>
                              <span className="flex items-center gap-1">
                                <Clock className="h-3 w-3" />
                                {interview.duration_minutes} minutes
                              </span>
                              <span>Interviewer: {interview.interviewer}</span>
                            </div>
                          </div>
                          <Button variant="outline" size="sm">
                            View Details
                          </Button>
                        </div>
                      </Card>
                    );
                  })
                )}
              </div>
            )}
          </>
        )}
      </div>

      {/* Modals */}
      {showCreateJobModal && (
        <CreateJobModal onClose={() => setShowCreateJobModal(false)} />
      )}
      {showCreateCandidateModal && (
        <CreateCandidateModal onClose={() => setShowCreateCandidateModal(false)} />
      )}
      {showCreateApplicationModal && (
        <CreateApplicationModal onClose={() => setShowCreateApplicationModal(false)} />
      )}
      {showCreateInterviewModal && (
        <CreateInterviewModal onClose={() => setShowCreateInterviewModal(false)} />
      )}
    </div>
  );
}

