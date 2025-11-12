/**
 * Export Utilities
 * 
 * Utility functions for exporting data to various formats (CSV, JSON)
 */

/**
 * Download a file with the given content
 */
function downloadFile(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Convert array of objects to CSV string
 */
function arrayToCSV(data: any[], headers?: string[]): string {
  if (data.length === 0) return '';

  // Get headers from first object if not provided
  const csvHeaders = headers || Object.keys(data[0]);
  
  // Create header row
  const headerRow = csvHeaders.join(',');
  
  // Create data rows
  const dataRows = data.map(item => {
    return csvHeaders.map(header => {
      const value = item[header];
      
      // Handle different value types
      if (value === null || value === undefined) {
        return '';
      }
      
      // Convert arrays and objects to JSON strings
      if (typeof value === 'object') {
        return `"${JSON.stringify(value).replace(/"/g, '""')}"`;
      }
      
      // Escape quotes and wrap in quotes if contains comma or newline
      const stringValue = String(value);
      if (stringValue.includes(',') || stringValue.includes('\n') || stringValue.includes('"')) {
        return `"${stringValue.replace(/"/g, '""')}"`;
      }
      
      return stringValue;
    }).join(',');
  });
  
  return [headerRow, ...dataRows].join('\n');
}

/**
 * Export data to CSV format
 */
export function exportToCSV(data: any[], filename: string, headers?: string[]) {
  const csv = arrayToCSV(data, headers);
  downloadFile(csv, filename, 'text/csv;charset=utf-8;');
}

/**
 * Export data to JSON format
 */
export function exportToJSON(data: any, filename: string) {
  const json = JSON.stringify(data, null, 2);
  downloadFile(json, filename, 'application/json');
}

/**
 * Export CRM contacts to CSV
 */
export function exportContactsToCSV(contacts: any[]) {
  const headers = [
    'contact_id',
    'first_name',
    'last_name',
    'email',
    'phone',
    'title',
    'company_id',
    'status',
    'lead_score',
    'tags',
    'last_contacted',
    'created_at',
    'updated_at',
  ];
  
  const timestamp = new Date().toISOString().split('T')[0];
  exportToCSV(contacts, `crm_contacts_${timestamp}.csv`, headers);
}

/**
 * Export CRM companies to CSV
 */
export function exportCompaniesToCSV(companies: any[]) {
  const headers = [
    'company_id',
    'name',
    'industry',
    'size',
    'website',
    'phone',
    'address',
    'city',
    'state',
    'country',
    'tags',
    'created_at',
    'updated_at',
  ];
  
  const timestamp = new Date().toISOString().split('T')[0];
  exportToCSV(companies, `crm_companies_${timestamp}.csv`, headers);
}

/**
 * Export CRM deals to CSV
 */
export function exportDealsToCSV(deals: any[]) {
  const headers = [
    'deal_id',
    'title',
    'contact_id',
    'company_id',
    'value',
    'currency',
    'stage',
    'probability',
    'expected_close_date',
    'tags',
    'created_at',
    'updated_at',
  ];
  
  const timestamp = new Date().toISOString().split('T')[0];
  exportToCSV(deals, `crm_deals_${timestamp}.csv`, headers);
}

/**
 * Export ATS jobs to CSV
 */
export function exportJobsToCSV(jobs: any[]) {
  const headers = [
    'job_id',
    'title',
    'description',
    'job_type',
    'experience_level',
    'location',
    'remote_allowed',
    'salary_min',
    'salary_max',
    'salary_currency',
    'department',
    'hiring_manager',
    'status',
    'application_count',
    'view_count',
    'tags',
    'created_at',
    'updated_at',
    'posted_at',
    'closed_at',
  ];
  
  const timestamp = new Date().toISOString().split('T')[0];
  exportToCSV(jobs, `ats_jobs_${timestamp}.csv`, headers);
}

/**
 * Export ATS candidates to CSV
 */
export function exportCandidatesToCSV(candidates: any[]) {
  const headers = [
    'candidate_id',
    'first_name',
    'last_name',
    'email',
    'phone',
    'location',
    'current_title',
    'current_company',
    'experience_years',
    'skills',
    'resume_url',
    'linkedin_url',
    'portfolio_url',
    'status',
    'tags',
    'created_at',
    'updated_at',
  ];
  
  const timestamp = new Date().toISOString().split('T')[0];
  exportToCSV(candidates, `ats_candidates_${timestamp}.csv`, headers);
}

/**
 * Export ATS applications to CSV
 */
export function exportApplicationsToCSV(applications: any[]) {
  const headers = [
    'application_id',
    'job_id',
    'candidate_id',
    'status',
    'applied_at',
    'source',
    'cover_letter',
    'resume_url',
    'rating',
    'notes',
    'created_at',
    'updated_at',
  ];
  
  const timestamp = new Date().toISOString().split('T')[0];
  exportToCSV(applications, `ats_applications_${timestamp}.csv`, headers);
}

/**
 * Export ATS interviews to CSV
 */
export function exportInterviewsToCSV(interviews: any[]) {
  const headers = [
    'interview_id',
    'application_id',
    'candidate_id',
    'job_id',
    'type',
    'scheduled_at',
    'duration_minutes',
    'interviewer',
    'location',
    'meeting_link',
    'status',
    'feedback',
    'rating',
    'created_at',
    'updated_at',
  ];
  
  const timestamp = new Date().toISOString().split('T')[0];
  exportToCSV(interviews, `ats_interviews_${timestamp}.csv`, headers);
}

