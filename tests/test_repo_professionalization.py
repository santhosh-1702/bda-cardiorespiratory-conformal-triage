"""
Test Suite: Repository Professionalization for Sharing
======================================================
Verifies:
  1. LICENSE file exists, contains the MIT License, and explicitly notes that the
     license covers code only (BIDMC data remains governed by PhysioNet terms).
  2. CITATION.cff exists at the repository root and contains valid machine-readable
     citation fields adhering to CFF 1.2.0.
  3. requirements.txt contains exact pinned versions (using '==') matching the
     tested environment, not loose version bounds.
  4. Data and Code Availability statements credit PhysioNet and the BIDMC dataset
     with the exact required citation pair:
       - Pimentel, M.A.F. et al. (IEEE TBME, 2016)
       - Goldberger, A.L. et al. (Circulation, 2000)
"""

import unittest
import re
from pathlib import Path

from config.settings import PROJECT_ROOT


class TestRepoProfessionalization(unittest.TestCase):

    def setUp(self):
        self.license_path = PROJECT_ROOT / "LICENSE"
        self.citation_path = PROJECT_ROOT / "CITATION.cff"
        self.reqs_path = PROJECT_ROOT / "requirements.txt"
        self.avail_path = PROJECT_ROOT / "DATA_AND_CODE_AVAILABILITY.md"
        self.readme_path = PROJECT_ROOT / "README.md"

    def test_license_exists_and_contains_governance_notice(self):
        """Verify LICENSE exists, specifies MIT, and explicitly covers code only."""
        self.assertTrue(self.license_path.exists(), "LICENSE file must exist at repo root")
        text = self.license_path.read_text(encoding="utf-8")

        # Must be MIT
        self.assertIn("MIT License", text)
        self.assertIn("Permission is hereby granted, free of charge", text)

        # Must explicitly state it covers code only and not the BIDMC data
        self.assertIn("covers the code only", text.lower())
        self.assertIn("physionet", text.lower())
        self.assertIn("bidmc", text.lower())

    def test_citation_cff_exists_and_valid(self):
        """Verify CITATION.cff exists at repo root with required CFF 1.2.0 fields."""
        self.assertTrue(self.citation_path.exists(), "CITATION.cff must exist at repo root")
        text = self.citation_path.read_text(encoding="utf-8")

        required_cff_fields = [
            "cff-version",
            "message",
            "title",
            "authors",
            "license: MIT",
            "keywords",
            "version",
        ]
        for field in required_cff_fields:
            self.assertIn(field, text, f"CITATION.cff missing required field/value: {field}")

        self.assertIn("1.2.0", text)
        self.assertIn("Karlen", text)

    def test_requirements_pinned_exact_versions(self):
        """Verify requirements.txt has exact pinned versions (==) for all dependencies."""
        self.assertTrue(self.reqs_path.exists(), "requirements.txt must exist at repo root")
        lines = self.reqs_path.read_text(encoding="utf-8").splitlines()

        pkg_lines = [
            line.strip() for line in lines
            if line.strip() and not line.strip().startswith("#")
        ]

        # Every dependency must be pinned with '=='
        for pkg_line in pkg_lines:
            self.assertIn(
                "==",
                pkg_line,
                f"Dependency '{pkg_line}' is not pinned with an exact version ('==')"
            )
            self.assertNotIn(">=", pkg_line)
            self.assertNotIn("<=", pkg_line)

        # Core required packages must be present
        pinned_text = "\n".join(pkg_lines).lower()
        for expected in ["numpy", "scipy", "pandas", "wfdb", "xgboost", "mapie", "matplotlib", "neurokit2"]:
            self.assertIn(expected, pinned_text, f"Required core package '{expected}' missing from requirements.txt")

    def test_data_and_code_availability_citations(self):
        """Verify availability document and README include both mandatory citations."""
        self.assertTrue(self.avail_path.exists(), "DATA_AND_CODE_AVAILABILITY.md must exist at repo root")
        avail_text = self.avail_path.read_text(encoding="utf-8")
        readme_text = self.readme_path.read_text(encoding="utf-8")

        # Pimentel et al. 2016 citation check
        self.assertIn("Pimentel", avail_text)
        self.assertIn("Towards a Robust Estimation of Respiratory Rate", avail_text)
        self.assertIn("1914", avail_text)
        self.assertIn("2016", avail_text)

        # Goldberger et al. 2000 citation check
        self.assertIn("Goldberger", avail_text)
        self.assertIn("PhysioBank, PhysioToolkit, and PhysioNet", avail_text)
        self.assertIn("Circulation", avail_text)
        self.assertIn("2000", avail_text)

        # Karlen et al. 2013 citation check
        self.assertIn("Karlen", avail_text)
        self.assertIn("Multiparameter Respiratory Rate Estimation", avail_text)
        self.assertIn("1946", avail_text)
        self.assertIn("2013", avail_text)

        # README section check
        self.assertIn("## 11. Data & Code Availability", readme_text)
        self.assertIn("PhysioNet", readme_text)
        self.assertIn("BIDMC Respiratory Rate Dataset", readme_text)


if __name__ == "__main__":
    unittest.main()
