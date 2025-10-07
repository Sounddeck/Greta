"""
GRETA CRM Service
Customer Relationship Management with AI-powered insights
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)

class Contact:
    def __init__(self, contact_data: Dict[str, Any]):
        self.id = contact_data.get('id', str(uuid.uuid4()))
        self.name = contact_data.get('name', '')
        self.email = contact_data.get('email', '')
        self.phone = contact_data.get('phone', '')
        self.company = contact_data.get('company', '')
        self.position = contact_data.get('position', '')
        self.status = contact_data.get('status', 'lead')  # lead, prospect, customer, inactive
        self.source = contact_data.get('source', 'manual')
        self.tags = contact_data.get('tags', [])
        self.notes = contact_data.get('notes', [])
        self.created_at = contact_data.get('created_at', datetime.now().isoformat())
        self.updated_at = contact_data.get('updated_at', datetime.now().isoformat())
        self.last_contact = contact_data.get('last_contact', None)
        self.value = contact_data.get('value', 0.0)
        self.probability = contact_data.get('probability', 0.0)

class Deal:
    def __init__(self, deal_data: Dict[str, Any]):
        self.id = deal_data.get('id', str(uuid.uuid4()))
        self.title = deal_data.get('title', '')
        self.contact_id = deal_data.get('contact_id', '')
        self.value = deal_data.get('value', 0.0)
        self.stage = deal_data.get('stage', 'qualification')  # qualification, proposal, negotiation, closed_won, closed_lost
        self.probability = deal_data.get('probability', 0.0)
        self.expected_close = deal_data.get('expected_close', None)
        self.created_at = deal_data.get('created_at', datetime.now().isoformat())
        self.updated_at = deal_data.get('updated_at', datetime.now().isoformat())
        self.notes = deal_data.get('notes', [])

class CRMService:
    """
    AI-powered Customer Relationship Management
    """
    
    def __init__(self):
        self.data_dir = Path("data/crm")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.contacts_file = self.data_dir / "contacts.json"
        self.deals_file = self.data_dir / "deals.json"
        self.activities_file = self.data_dir / "activities.json"
        
        self.contacts = self._load_contacts()
        self.deals = self._load_deals()
        self.activities = self._load_activities()
        
        logger.info("✅ CRM Service initialized")
    
    def _load_contacts(self) -> Dict[str, Contact]:
        """Load contacts from file"""
        if self.contacts_file.exists():
            try:
                with open(self.contacts_file, 'r') as f:
                    data = json.load(f)
                return {cid: Contact(cdata) for cid, cdata in data.items()}
            except Exception as e:
                logger.error(f"Error loading contacts: {e}")
        return {}
    
    def _load_deals(self) -> Dict[str, Deal]:
        """Load deals from file"""
        if self.deals_file.exists():
            try:
                with open(self.deals_file, 'r') as f:
                    data = json.load(f)
                return {did: Deal(ddata) for did, ddata in data.items()}
            except Exception as e:
                logger.error(f"Error loading deals: {e}")
        return {}
    
    def _load_activities(self) -> List[Dict[str, Any]]:
        """Load activities from file"""
        if self.activities_file.exists():
            try:
                with open(self.activities_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading activities: {e}")
        return []
    
    def _save_contacts(self):
        """Save contacts to file"""
        try:
            data = {cid: contact.__dict__ for cid, contact in self.contacts.items()}
            with open(self.contacts_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving contacts: {e}")
    
    def _save_deals(self):
        """Save deals to file"""
        try:
            data = {did: deal.__dict__ for did, deal in self.deals.items()}
            with open(self.deals_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving deals: {e}")
    
    def _save_activities(self):
        """Save activities to file"""
        try:
            with open(self.activities_file, 'w') as f:
                json.dump(self.activities, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving activities: {e}")
    
    async def create_contact(self, contact_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new contact"""
        try:
            contact = Contact(contact_data)
            self.contacts[contact.id] = contact
            self._save_contacts()
            
            # Log activity
            await self._log_activity({
                "type": "contact_created",
                "contact_id": contact.id,
                "description": f"New contact created: {contact.name}",
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "success": True,
                "contact_id": contact.id,
                "message": f"Contact {contact.name} created successfully"
            }
        except Exception as e:
            logger.error(f"Error creating contact: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_contacts(self, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get contacts with optional filters"""
        try:
            contacts_list = []
            for contact in self.contacts.values():
                contact_dict = contact.__dict__.copy()
                
                # Apply filters if provided
                if filters:
                    if filters.get('status') and contact.status != filters['status']:
                        continue
                    if filters.get('company') and filters['company'].lower() not in contact.company.lower():
                        continue
                    if filters.get('search'):
                        search_term = filters['search'].lower()
                        if not any(search_term in str(getattr(contact, field, '')).lower() 
                                 for field in ['name', 'email', 'company', 'position']):
                            continue
                
                contacts_list.append(contact_dict)
            
            return {
                "success": True,
                "contacts": contacts_list,
                "total": len(contacts_list)
            }
        except Exception as e:
            logger.error(f"Error getting contacts: {e}")
            return {"success": False, "error": str(e)}
    
    async def create_deal(self, deal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new deal"""
        try:
            deal = Deal(deal_data)
            self.deals[deal.id] = deal
            self._save_deals()
            
            # Log activity
            await self._log_activity({
                "type": "deal_created",
                "deal_id": deal.id,
                "contact_id": deal.contact_id,
                "description": f"New deal created: {deal.title} (${deal.value})",
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "success": True,
                "deal_id": deal.id,
                "message": f"Deal {deal.title} created successfully"
            }
        except Exception as e:
            logger.error(f"Error creating deal: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_deals(self, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get deals with optional filters"""
        try:
            deals_list = []
            for deal in self.deals.values():
                deal_dict = deal.__dict__.copy()
                
                # Add contact information
                if deal.contact_id in self.contacts:
                    contact = self.contacts[deal.contact_id]
                    deal_dict['contact_name'] = contact.name
                    deal_dict['contact_company'] = contact.company
                
                # Apply filters if provided
                if filters:
                    if filters.get('stage') and deal.stage != filters['stage']:
                        continue
                    if filters.get('contact_id') and deal.contact_id != filters['contact_id']:
                        continue
                
                deals_list.append(deal_dict)
            
            return {
                "success": True,
                "deals": deals_list,
                "total": len(deals_list)
            }
        except Exception as e:
            logger.error(f"Error getting deals: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_analytics(self) -> Dict[str, Any]:
        """Get CRM analytics and insights"""
        try:
            total_contacts = len(self.contacts)
            total_deals = len(self.deals)
            
            # Deal stage distribution
            stage_counts = {}
            total_value = 0
            weighted_value = 0
            
            for deal in self.deals.values():
                stage_counts[deal.stage] = stage_counts.get(deal.stage, 0) + 1
                total_value += deal.value
                weighted_value += deal.value * (deal.probability / 100)
            
            # Contact status distribution
            status_counts = {}
            for contact in self.contacts.values():
                status_counts[contact.status] = status_counts.get(contact.status, 0) + 1
            
            # Recent activities
            recent_activities = sorted(
                self.activities[-10:], 
                key=lambda x: x.get('timestamp', ''), 
                reverse=True
            )
            
            return {
                "success": True,
                "analytics": {
                    "total_contacts": total_contacts,
                    "total_deals": total_deals,
                    "total_deal_value": total_value,
                    "weighted_deal_value": weighted_value,
                    "deal_stages": stage_counts,
                    "contact_statuses": status_counts,
                    "recent_activities": recent_activities,
                    "conversion_rate": self._calculate_conversion_rate(),
                    "avg_deal_size": total_value / max(total_deals, 1)
                }
            }
        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            return {"success": False, "error": str(e)}
    
    def _calculate_conversion_rate(self) -> float:
        """Calculate lead to customer conversion rate"""
        try:
            leads = sum(1 for c in self.contacts.values() if c.status == 'lead')
            customers = sum(1 for c in self.contacts.values() if c.status == 'customer')
            
            if leads + customers == 0:
                return 0.0
            
            return (customers / (leads + customers)) * 100
        except:
            return 0.0
    
    async def _log_activity(self, activity: Dict[str, Any]):
        """Log an activity"""
        self.activities.append(activity)
        # Keep only last 1000 activities
        if len(self.activities) > 1000:
            self.activities = self.activities[-1000:]
        self._save_activities()
    
    async def ai_insights(self, contact_id: str = None) -> Dict[str, Any]:
        """Generate AI-powered insights"""
        try:
            insights = []
            
            # General insights
            if not contact_id:
                # Pipeline health
                open_deals = [d for d in self.deals.values() if d.stage not in ['closed_won', 'closed_lost']]
                if len(open_deals) > 0:
                    avg_deal_age = sum((datetime.now() - datetime.fromisoformat(d.created_at.replace('Z', '+00:00'))).days for d in open_deals) / len(open_deals)
                    if avg_deal_age > 30:
                        insights.append({
                            "type": "warning",
                            "title": "Aging Pipeline",
                            "description": f"Average deal age is {avg_deal_age:.0f} days. Consider following up on stale opportunities."
                        })
                
                # Contact engagement
                inactive_contacts = [c for c in self.contacts.values() 
                                   if not c.last_contact or 
                                   (datetime.now() - datetime.fromisoformat(c.last_contact.replace('Z', '+00:00'))).days > 30]
                
                if len(inactive_contacts) > 5:
                    insights.append({
                        "type": "opportunity",
                        "title": "Re-engagement Opportunity",
                        "description": f"{len(inactive_contacts)} contacts haven't been contacted in 30+ days."
                    })
            
            # Contact-specific insights
            else:
                if contact_id in self.contacts:
                    contact = self.contacts[contact_id]
                    contact_deals = [d for d in self.deals.values() if d.contact_id == contact_id]
                    
                    if len(contact_deals) == 0:
                        insights.append({
                            "type": "opportunity",
                            "title": "Deal Opportunity",
                            "description": f"No deals associated with {contact.name}. Consider creating an opportunity."
                        })
                    
                    if contact.last_contact:
                        days_since_contact = (datetime.now() - datetime.fromisoformat(contact.last_contact.replace('Z', '+00:00'))).days
                        if days_since_contact > 14:
                            insights.append({
                                "type": "action",
                                "title": "Follow-up Needed",
                                "description": f"Last contact was {days_since_contact} days ago."
                            })
            
            return {
                "success": True,
                "insights": insights
            }
            
        except Exception as e:
            logger.error(f"Error generating AI insights: {e}")
            return {"success": False, "error": str(e)}

# Global instance
crm_service = CRMService()
