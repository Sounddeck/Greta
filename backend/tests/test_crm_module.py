"""
CRM Module - Comprehensive Test Suite
Tests all CRM functionality with real database integration
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import Database
from services.crm_service_new import CRMService
from models.crm_models import (
    ContactCreate, ContactUpdate, ContactStatus,
    CompanyCreate, CompanyUpdate, CompanySize,
    DealCreate, DealUpdate, DealStage,
    ActivityCreate, ActivityUpdate, ActivityType, Priority
)
from datetime import datetime, timedelta


async def test_contact_crud():
    """Test 1: Contact CRUD Operations"""
    print("\n" + "="*80)
    print("TEST 1: Contact CRUD Operations")
    print("="*80)
    
    try:
        db = Database()
        await db.connect()
        service = CRMService(db)
        
        # Create contact
        print("\n📝 Test 1.1: Create Contact")
        contact_create = ContactCreate(
            first_name="John",
            last_name="Doe",
            email="john.doe@example.com",
            phone="+1-555-0123",
            title="CEO",
            status=ContactStatus.LEAD,
            tags=["vip", "tech"]
        )
        
        contact = await service.create_contact(contact_create)
        print(f"✅ Contact created: {contact.contact_id}")
        print(f"   Name: {contact.first_name} {contact.last_name}")
        print(f"   Email: {contact.email}")
        print(f"   Status: {contact.status}")
        
        contact_id = contact.contact_id
        
        # Get contact
        print("\n📖 Test 1.2: Get Contact")
        retrieved = await service.get_contact(contact_id)
        assert retrieved is not None, "Contact not found"
        assert retrieved.email == "john.doe@example.com", "Email mismatch"
        print(f"✅ Contact retrieved: {retrieved.contact_id}")
        
        # Update contact
        print("\n✏️  Test 1.3: Update Contact")
        update = ContactUpdate(
            status=ContactStatus.PROSPECT,
            lead_score=75.5
        )
        updated = await service.update_contact(contact_id, update)
        assert updated.status == ContactStatus.PROSPECT, "Status not updated"
        assert updated.lead_score == 75.5, "Lead score not updated"
        print(f"✅ Contact updated: {updated.status}, Score: {updated.lead_score}")
        
        # List contacts
        print("\n📋 Test 1.4: List Contacts")
        contacts = await service.list_contacts(status=ContactStatus.PROSPECT)
        assert len(contacts) > 0, "No contacts found"
        print(f"✅ Found {len(contacts)} prospect contacts")
        
        # Delete contact
        print("\n🗑️  Test 1.5: Delete Contact")
        deleted = await service.delete_contact(contact_id)
        assert deleted, "Contact not deleted"
        
        # Verify deletion
        verify = await service.get_contact(contact_id)
        assert verify is None, "Contact still exists"
        print(f"✅ Contact deleted: {contact_id}")
        
        await db.disconnect()
        print("\n✅ Contact CRUD test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Contact CRUD test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_company_crud():
    """Test 2: Company CRUD Operations"""
    print("\n" + "="*80)
    print("TEST 2: Company CRUD Operations")
    print("="*80)
    
    try:
        db = Database()
        await db.connect()
        service = CRMService(db)
        
        # Create company
        print("\n📝 Test 2.1: Create Company")
        company_create = CompanyCreate(
            name="Acme Corporation",
            industry="Technology",
            size=CompanySize.MEDIUM,
            website="https://acme.com",
            city="San Francisco",
            state="CA",
            country="USA",
            tags=["saas", "b2b"]
        )
        
        company = await service.create_company(company_create)
        print(f"✅ Company created: {company.company_id}")
        print(f"   Name: {company.name}")
        print(f"   Industry: {company.industry}")
        print(f"   Size: {company.size}")
        
        company_id = company.company_id
        
        # Get company
        print("\n📖 Test 2.2: Get Company")
        retrieved = await service.get_company(company_id)
        assert retrieved is not None, "Company not found"
        assert retrieved.name == "Acme Corporation", "Name mismatch"
        print(f"✅ Company retrieved: {retrieved.company_id}")
        
        # Update company
        print("\n✏️  Test 2.3: Update Company")
        update = CompanyUpdate(
            size=CompanySize.LARGE,
            phone="+1-555-9999"
        )
        updated = await service.update_company(company_id, update)
        assert updated.size == CompanySize.LARGE, "Size not updated"
        print(f"✅ Company updated: {updated.size}")
        
        # List companies
        print("\n📋 Test 2.4: List Companies")
        companies = await service.list_companies(industry="Technology")
        assert len(companies) > 0, "No companies found"
        print(f"✅ Found {len(companies)} technology companies")
        
        # Delete company
        print("\n🗑️  Test 2.5: Delete Company")
        deleted = await service.delete_company(company_id)
        assert deleted, "Company not deleted"
        print(f"✅ Company deleted: {company_id}")
        
        await db.disconnect()
        print("\n✅ Company CRUD test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Company CRUD test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_deal_crud():
    """Test 3: Deal CRUD Operations"""
    print("\n" + "="*80)
    print("TEST 3: Deal CRUD Operations")
    print("="*80)
    
    try:
        db = Database()
        await db.connect()
        service = CRMService(db)
        
        # Create deal
        print("\n📝 Test 3.1: Create Deal")
        deal_create = DealCreate(
            title="Enterprise Software License",
            description="Annual license for 100 users",
            value=50000.0,
            stage=DealStage.PROPOSAL,
            probability=0.7,
            expected_close_date=datetime.utcnow() + timedelta(days=30),
            tags=["enterprise", "software"]
        )
        
        deal = await service.create_deal(deal_create)
        print(f"✅ Deal created: {deal.deal_id}")
        print(f"   Title: {deal.title}")
        print(f"   Value: ${deal.value:,.2f}")
        print(f"   Stage: {deal.stage}")
        
        deal_id = deal.deal_id
        
        # Get deal
        print("\n📖 Test 3.2: Get Deal")
        retrieved = await service.get_deal(deal_id)
        assert retrieved is not None, "Deal not found"
        assert retrieved.value == 50000.0, "Value mismatch"
        print(f"✅ Deal retrieved: {retrieved.deal_id}")
        
        # Update deal
        print("\n✏️  Test 3.3: Update Deal")
        update = DealUpdate(
            stage=DealStage.NEGOTIATION,
            probability=0.85
        )
        updated = await service.update_deal(deal_id, update)
        assert updated.stage == DealStage.NEGOTIATION, "Stage not updated"
        print(f"✅ Deal updated: {updated.stage}, Probability: {updated.probability}")
        
        # List deals
        print("\n📋 Test 3.4: List Deals")
        deals = await service.list_deals(stage=DealStage.NEGOTIATION)
        assert len(deals) > 0, "No deals found"
        print(f"✅ Found {len(deals)} deals in negotiation")
        
        # Delete deal
        print("\n🗑️  Test 3.5: Delete Deal")
        deleted = await service.delete_deal(deal_id)
        assert deleted, "Deal not deleted"
        print(f"✅ Deal deleted: {deal_id}")
        
        await db.disconnect()
        print("\n✅ Deal CRUD test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Deal CRUD test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_activity_crud():
    """Test 4: Activity CRUD Operations"""
    print("\n" + "="*80)
    print("TEST 4: Activity CRUD Operations")
    print("="*80)
    
    try:
        db = Database()
        await db.connect()
        service = CRMService(db)
        
        # Create activity
        print("\n📝 Test 4.1: Create Activity")
        activity_create = ActivityCreate(
            activity_type=ActivityType.CALL,
            subject="Follow-up call with prospect",
            description="Discuss product features and pricing",
            due_date=datetime.utcnow() + timedelta(days=2),
            priority=Priority.HIGH,
            completed=False,
            tags=["follow-up", "sales"]
        )
        
        activity = await service.create_activity(activity_create)
        print(f"✅ Activity created: {activity.activity_id}")
        print(f"   Type: {activity.activity_type}")
        print(f"   Subject: {activity.subject}")
        print(f"   Priority: {activity.priority}")
        
        activity_id = activity.activity_id
        
        # Get activity
        print("\n📖 Test 4.2: Get Activity")
        retrieved = await service.get_activity(activity_id)
        assert retrieved is not None, "Activity not found"
        assert retrieved.activity_type == ActivityType.CALL, "Type mismatch"
        print(f"✅ Activity retrieved: {retrieved.activity_id}")
        
        # Update activity - mark as completed
        print("\n✏️  Test 4.3: Update Activity (Complete)")
        update = ActivityUpdate(completed=True)
        updated = await service.update_activity(activity_id, update)
        assert updated.completed == True, "Not marked as completed"
        assert updated.completed_at is not None, "Completed timestamp not set"
        print(f"✅ Activity completed: {updated.completed_at}")
        
        # List activities
        print("\n📋 Test 4.4: List Activities")
        activities = await service.list_activities(completed=True)
        assert len(activities) > 0, "No activities found"
        print(f"✅ Found {len(activities)} completed activities")
        
        # Delete activity
        print("\n🗑️  Test 4.5: Delete Activity")
        deleted = await service.delete_activity(activity_id)
        assert deleted, "Activity not deleted"
        print(f"✅ Activity deleted: {activity_id}")
        
        await db.disconnect()
        print("\n✅ Activity CRUD test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Activity CRUD test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_ai_features():
    """Test 5: AI-Powered Features"""
    print("\n" + "="*80)
    print("TEST 5: AI-Powered Features")
    print("="*80)

    try:
        db = Database()
        await db.connect()
        service = CRMService(db)

        # Create test contact
        print("\n📝 Test 5.1: Create Test Contact for AI")
        contact_create = ContactCreate(
            first_name="Jane",
            last_name="Smith",
            email="jane.smith@example.com",
            phone="+1-555-0456",
            title="CTO",
            status=ContactStatus.LEAD,
            tags=["tech", "decision-maker"]
        )
        contact = await service.create_contact(contact_create)
        contact_id = contact.contact_id
        print(f"✅ Test contact created: {contact_id}")

        # Create some activities for the contact
        print("\n📝 Test 5.2: Create Activities for Engagement")
        for i in range(3):
            activity_create = ActivityCreate(
                activity_type=ActivityType.EMAIL,
                subject=f"Email {i+1}",
                description=f"Test email {i+1}",
                contact_id=contact_id,
                completed=True
            )
            await service.create_activity(activity_create)
        print(f"✅ Created 3 activities for contact")

        # Calculate lead score
        print("\n🤖 Test 5.3: Calculate AI Lead Score")
        lead_score = await service.calculate_lead_score(contact_id)
        print(f"✅ Lead score calculated: {lead_score.score:.1f}/100")
        print(f"   Confidence: {lead_score.confidence:.2f}")
        print(f"   Factors:")
        for factor, value in lead_score.factors.items():
            print(f"     - {factor}: {value:.1f}")
        print(f"   Recommendations: {len(lead_score.recommendations)}")
        for rec in lead_score.recommendations:
            print(f"     - {rec}")

        assert lead_score.score > 0, "Lead score should be positive"
        assert lead_score.confidence > 0, "Confidence should be positive"

        # Suggest next action
        print("\n🤖 Test 5.4: Suggest Next Action")
        next_action = await service.suggest_next_action("contact", contact_id)
        print(f"✅ Next action suggested:")
        print(f"   Type: {next_action.action_type}")
        print(f"   Subject: {next_action.subject}")
        print(f"   Priority: {next_action.priority}")
        print(f"   Reasoning: {next_action.reasoning}")
        print(f"   Confidence: {next_action.confidence:.2f}")

        assert next_action.action_type is not None, "Action type should be set"
        assert next_action.confidence > 0, "Confidence should be positive"

        # Cleanup
        await service.delete_contact(contact_id)

        await db.disconnect()
        print("\n✅ AI Features test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ AI Features test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_crm_statistics():
    """Test 6: CRM Statistics"""
    print("\n" + "="*80)
    print("TEST 6: CRM Statistics")
    print("="*80)

    try:
        db = Database()
        await db.connect()
        service = CRMService(db)

        # Create test data
        print("\n📝 Creating test data...")

        # Create contacts
        for i in range(5):
            contact_create = ContactCreate(
                first_name=f"Contact{i}",
                last_name="Test",
                email=f"contact{i}@test.com",
                status=[ContactStatus.LEAD, ContactStatus.PROSPECT, ContactStatus.CUSTOMER][i % 3]
            )
            await service.create_contact(contact_create)

        # Create companies
        for i in range(3):
            company_create = CompanyCreate(
                name=f"Company {i}",
                industry="Technology"
            )
            await service.create_company(company_create)

        # Create deals
        for i in range(4):
            deal_create = DealCreate(
                title=f"Deal {i}",
                value=10000.0 * (i + 1),
                stage=[DealStage.LEAD, DealStage.PROPOSAL, DealStage.CLOSED_WON, DealStage.CLOSED_LOST][i]
            )
            await service.create_deal(deal_create)

        # Create activities
        for i in range(6):
            activity_create = ActivityCreate(
                activity_type=[ActivityType.CALL, ActivityType.EMAIL, ActivityType.MEETING][i % 3],
                subject=f"Activity {i}",
                completed=i % 2 == 0
            )
            await service.create_activity(activity_create)

        print(f"✅ Created test data: 5 contacts, 3 companies, 4 deals, 6 activities")

        # Get statistics
        print("\n📊 Test 6.1: Get CRM Statistics")
        stats = await service.get_crm_stats()

        print(f"✅ CRM Statistics:")
        print(f"   Total Contacts: {stats.total_contacts}")
        print(f"   Total Companies: {stats.total_companies}")
        print(f"   Total Deals: {stats.total_deals}")
        print(f"   Total Activities: {stats.total_activities}")
        print(f"\n   Contacts by Status:")
        for status, count in stats.contacts_by_status.items():
            print(f"     - {status}: {count}")
        print(f"\n   Deals by Stage:")
        for stage, count in stats.deals_by_stage.items():
            print(f"     - {stage}: {count}")
        print(f"\n   Pipeline Value: ${stats.total_pipeline_value:,.2f}")
        print(f"   Won Value: ${stats.total_won_value:,.2f}")
        print(f"   Average Deal Size: ${stats.average_deal_size:,.2f}")
        print(f"   Win Rate: {stats.win_rate:.1f}%")

        assert stats.total_contacts >= 5, "Should have at least 5 contacts"
        assert stats.total_companies >= 3, "Should have at least 3 companies"
        assert stats.total_deals >= 4, "Should have at least 4 deals"
        assert stats.total_activities >= 6, "Should have at least 6 activities"

        await db.disconnect()
        print("\n✅ CRM Statistics test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ CRM Statistics test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all CRM tests"""
    print("\n" + "="*80)
    print("CPAS4 CRM MODULE - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print("\nTesting:")
    print("1. Contact CRUD Operations")
    print("2. Company CRUD Operations")
    print("3. Deal CRUD Operations")
    print("4. Activity CRUD Operations")
    print("5. AI-Powered Features (Lead Scoring, Next Actions)")
    print("6. CRM Statistics and Analytics")
    print("\n" + "="*80)

    results = []

    # Run tests
    results.append(await test_contact_crud())
    results.append(await test_company_crud())
    results.append(await test_deal_crud())
    results.append(await test_activity_crud())
    results.append(await test_ai_features())
    results.append(await test_crm_statistics())

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(results)
    total = len(results)

    print(f"\nTests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! CRM module is fully functional.")
        print("\n✅ Contact Management: WORKING")
        print("✅ Company Management: WORKING")
        print("✅ Deal Pipeline: WORKING")
        print("✅ Activity Tracking: WORKING")
        print("✅ AI Lead Scoring: WORKING")
        print("✅ AI Next Actions: WORKING")
        print("✅ Statistics: WORKING")
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")

    print("\n" + "="*80)


if __name__ == "__main__":
    asyncio.run(main())


