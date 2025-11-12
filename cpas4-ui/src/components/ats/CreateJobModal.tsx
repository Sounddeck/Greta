/**
 * Create Job Modal
 * 
 * Modal for creating a new ATS job posting
 */

import React, { useState } from 'react';
import { X } from 'lucide-react';
import { useATSStore } from '../../stores/atsStore';
import { toast } from 'sonner';

interface CreateJobModalProps {
  onClose: () => void;
}

export default function CreateJobModal({ onClose }: CreateJobModalProps) {
  const { createJob } = useATSStore();
  const [loading, setLoading] = useState(false);

  const [formData, setFormData] = useState({
    title: '',
    description: '',
    requirements: [] as string[],
    responsibilities: [] as string[],
    skills_required: [] as string[],
    skills_preferred: [] as string[],
    job_type: 'full_time' as 'full_time' | 'part_time' | 'contract' | 'temporary' | 'internship',
    experience_level: 'mid' as 'entry' | 'junior' | 'mid' | 'senior' | 'lead' | 'executive',
    location: '',
    remote_allowed: false,
    salary_min: 0,
    salary_max: 0,
    salary_currency: 'USD',
    department: '',
    hiring_manager: '',
    status: 'draft' as 'draft' | 'open' | 'closed' | 'on_hold' | 'filled',
    tags: [] as string[],
  });

  const [requirementInput, setRequirementInput] = useState('');
  const [responsibilityInput, setResponsibilityInput] = useState('');
  const [skillRequiredInput, setSkillRequiredInput] = useState('');
  const [tagInput, setTagInput] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      await createJob(formData);
      toast.success('Job created successfully');
      onClose();
    } catch (error) {
      toast.error('Failed to create job');
    } finally {
      setLoading(false);
    }
  };

  const handleAddRequirement = () => {
    if (requirementInput.trim() && !formData.requirements.includes(requirementInput.trim())) {
      setFormData({
        ...formData,
        requirements: [...formData.requirements, requirementInput.trim()],
      });
      setRequirementInput('');
    }
  };

  const handleRemoveRequirement = (req: string) => {
    setFormData({
      ...formData,
      requirements: formData.requirements.filter(r => r !== req),
    });
  };

  const handleAddResponsibility = () => {
    if (responsibilityInput.trim() && !formData.responsibilities.includes(responsibilityInput.trim())) {
      setFormData({
        ...formData,
        responsibilities: [...formData.responsibilities, responsibilityInput.trim()],
      });
      setResponsibilityInput('');
    }
  };

  const handleRemoveResponsibility = (resp: string) => {
    setFormData({
      ...formData,
      responsibilities: formData.responsibilities.filter(r => r !== resp),
    });
  };

  const handleAddSkillRequired = () => {
    if (skillRequiredInput.trim() && !formData.skills_required.includes(skillRequiredInput.trim())) {
      setFormData({
        ...formData,
        skills_required: [...formData.skills_required, skillRequiredInput.trim()],
      });
      setSkillRequiredInput('');
    }
  };

  const handleRemoveSkillRequired = (skill: string) => {
    setFormData({
      ...formData,
      skills_required: formData.skills_required.filter(s => s !== skill),
    });
  };

  const handleAddTag = () => {
    if (tagInput.trim() && !formData.tags.includes(tagInput.trim())) {
      setFormData({
        ...formData,
        tags: [...formData.tags, tagInput.trim()],
      });
      setTagInput('');
    }
  };

  const handleRemoveTag = (tag: string) => {
    setFormData({
      ...formData,
      tags: formData.tags.filter(t => t !== tag),
    });
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
            Create New Job Posting
          </h2>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Job Title *
            </label>
            <input
              type="text"
              required
              value={formData.title}
              onChange={(e) => setFormData({ ...formData, title: e.target.value })}
              className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-gray-100"
              placeholder="Senior Software Engineer"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Description *
            </label>
            <textarea
              required
              rows={4}
              value={formData.description}
              onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-gray-100"
              placeholder="Describe the role..."
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Job Type *
              </label>
              <select
                required
                value={formData.job_type}
                onChange={(e) => setFormData({ ...formData, job_type: e.target.value as any })}
                className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-gray-100"
              >
                <option value="full_time">Full Time</option>
                <option value="part_time">Part Time</option>
                <option value="contract">Contract</option>
                <option value="temporary">Temporary</option>
                <option value="internship">Internship</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Experience Level *
              </label>
              <select
                required
                value={formData.experience_level}
                onChange={(e) => setFormData({ ...formData, experience_level: e.target.value as any })}
                className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-gray-100"
              >
                <option value="entry">Entry Level</option>
                <option value="junior">Junior</option>
                <option value="mid">Mid Level</option>
                <option value="senior">Senior</option>
                <option value="lead">Lead</option>
                <option value="executive">Executive</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Location *
              </label>
              <input
                type="text"
                required
                value={formData.location}
                onChange={(e) => setFormData({ ...formData, location: e.target.value })}
                className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-gray-100"
                placeholder="San Francisco, CA"
              />
            </div>

            <div className="flex items-center pt-6">
              <input
                type="checkbox"
                id="remote_allowed"
                checked={formData.remote_allowed}
                onChange={(e) => setFormData({ ...formData, remote_allowed: e.target.checked })}
                className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500"
              />
              <label htmlFor="remote_allowed" className="ml-2 text-sm font-medium text-gray-700 dark:text-gray-300">
                Remote Allowed
              </label>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Min Salary
              </label>
              <input
                type="number"
                min="0"
                value={formData.salary_min}
                onChange={(e) => setFormData({ ...formData, salary_min: parseInt(e.target.value) || 0 })}
                className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-gray-100"
                placeholder="80000"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Max Salary
              </label>
              <input
                type="number"
                min="0"
                value={formData.salary_max}
                onChange={(e) => setFormData({ ...formData, salary_max: parseInt(e.target.value) || 0 })}
                className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-gray-100"
                placeholder="120000"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Currency
              </label>
              <select
                value={formData.salary_currency}
                onChange={(e) => setFormData({ ...formData, salary_currency: e.target.value })}
                className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg text-gray-900 dark:text-gray-100"
              >
                <option value="USD">USD</option>
                <option value="EUR">EUR</option>
                <option value="GBP">GBP</option>
              </select>
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-3 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? 'Creating...' : 'Create Job'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

