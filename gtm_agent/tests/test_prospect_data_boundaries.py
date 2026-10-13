import json
import unittest
from unittest.mock import Mock, patch

from gtm_agent import gtm_agent


class ProspectDataBoundaryTests(unittest.TestCase):
    def test_get_prospect_returns_only_contact_and_qualification_fields(self):
        record = {
            "name": "Jordan Smith",
            "email": "jordan@example.com",
            "disqualified": True,
            "annual_revenue": 5000000,
            "billing_qualification": {"test_marker": "excluded"},
        }
        with patch.object(gtm_agent.data_service, "get_prospect_record", return_value=record):
            result = gtm_agent.get_prospect.func(prospect_id="prospect-1")

        self.assertEqual(
            result,
            {
                "prospect": {
                    "prospect_id": "prospect-1",
                    "name": "Jordan Smith",
                    "email": "jordan@example.com",
                    "disqualified": True,
                },
                "found": True,
            },
        )

    def test_get_prospect_preserves_missing_optional_fields_and_defaults_status(self):
        with patch.object(gtm_agent.data_service, "get_prospect_record", return_value={}):
            result = gtm_agent.get_prospect.func(prospect_id="prospect-1")

        self.assertEqual(
            result,
            {"prospect": {"prospect_id": "prospect-1", "disqualified": False}, "found": True},
        )

    def test_new_profile_is_allowlisted_before_caching_and_returning(self):
        record = {
            "name": "Jordan Smith",
            "email": "jordan@example.com",
            "annual_revenue": 5000000,
            "enrichment_source": "crm",
            "billing_qualification": {"test_marker": "excluded"},
            "disqualified": True,
        }
        with (
            patch.object(gtm_agent.data_service, "get_profile_from_db", return_value={"prospect_profile": None}),
            patch.object(gtm_agent.data_service, "get_prospect_record", return_value=record),
            patch.object(gtm_agent.data_service, "fetch_engagement_history", return_value=["call"]),
            patch.object(gtm_agent.data_service, "fetch_account_details", return_value={"industry": "software"}),
            patch.object(gtm_agent.data_service, "fetch_tech_stack", return_value=["CRM"]),
            patch.object(gtm_agent.data_service, "save_profile_to_db") as save_profile,
        ):
            result = gtm_agent.build_prospect_profile.func(prospect_id="prospect-1")

        expected = {
            "prospect_id": "prospect-1",
            "name": "Jordan Smith",
            "email": "jordan@example.com",
            "annual_revenue": 5000000,
            "enrichment_source": "crm",
            "engagement_history": ["call"],
            "account_details": {"industry": "software"},
            "tech_stack": ["CRM"],
        }
        self.assertEqual(result, {"prospect_profile": expected, "found": True})
        save_profile.assert_called_once_with("prospect-1", expected)

    def test_cached_profile_is_projected_without_mutating_cache_entry(self):
        cached = {
            "prospect_id": "prospect-1",
            "name": "Jordan Smith",
            "email": "jordan@example.com",
            "annual_revenue": 5000000,
            "enrichment_source": "crm",
            "engagement_history": ["call"],
            "account_details": {"industry": "software"},
            "tech_stack": ["CRM"],
            "billing_qualification": {"test_marker": "excluded"},
        }
        with patch.object(
            gtm_agent.data_service,
            "get_profile_from_db",
            return_value={"prospect_profile": cached},
        ):
            result = gtm_agent.build_prospect_profile.func(prospect_id="prospect-1")

        self.assertEqual(
            set(result["prospect_profile"]),
            {
                "prospect_id", "name", "email", "annual_revenue", "enrichment_source",
                "engagement_history", "account_details", "tech_stack",
            },
        )
        self.assertIn("billing_qualification", cached)

    def test_scoring_input_projects_profile_and_refreshes_tech_stack(self):
        profile = {
            "prospect_id": "prospect-1",
            "name": "Jordan Smith",
            "annual_revenue": 5000000,
            "tech_stack": ["stale"],
            "account_details": {"industry": "software"},
            "email": "jordan@example.com",
            "billing_qualification": {"test_marker": "excluded"},
        }
        offering = {
            "required_tech_stack": ["CRM"],
            "min_annual_revenue": 1000000,
            "description": "A test offering",
        }
        llm_result = Mock()
        llm_result.model_dump.return_value = {"score": 80}
        scoring_llm = Mock(invoke=Mock(return_value=llm_result))
        with (
            patch.object(gtm_agent.data_service, "fetch_tech_stack", return_value=["CRM"]),
            patch.object(gtm_agent, "_scoring_llm", scoring_llm),
        ):
            gtm_agent.score_prospect.func(prospect_profile=profile, offering=offering)

        messages = scoring_llm.invoke.call_args.args[0]
        user_content = messages[1]["content"]
        serialized_profile = user_content.split("\n\nProspect profile:\n", 1)[1]
        self.assertEqual(
            json.loads(serialized_profile),
            {
                "prospect_id": "prospect-1",
                "name": "Jordan Smith",
                "annual_revenue": 5000000,
                "tech_stack": ["CRM"],
                "account_details": {"industry": "software"},
            },
        )


if __name__ == "__main__":
    unittest.main()
