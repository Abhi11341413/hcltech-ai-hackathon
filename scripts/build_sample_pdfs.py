"""Regenerate the SYNTHETIC PDF documents in data/sample_docs/.

Only needed if you change the content below. Requires: pip install fpdf2
    python -m scripts.build_sample_pdfs

Each PDF has a title, numbered section headings (bold) and body text, and some
sections start on a new page so the loader's page metadata can be tested.
Orbitly Inc. is a fictional company.
"""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"
NOTICE = "SYNTHETIC sample document created by the team for development and testing. Orbitly Inc. is a fictional company."

# (filename, title, meta line, [(heading, body, new_page)])
DOCUMENTS: list[tuple[str, str, str, list[tuple[str, str, bool]]]] = [
    (
        "sso_admin_guide_v3.pdf",
        "Single Sign-On Administration Guide (v3.0)",
        "Applies to version 3.0. Effective from 1 October 2026.",
        [
            ("1. Overview",
             "Orbitly Workspace supports SAML 2.0 single sign-on with Okta, Microsoft Entra ID (formerly Azure AD) "
             "and Google Workspace. SSO is available on the Enterprise plan only. Only Owners and Admins can configure SSO.",
             False),
            ("2. Configuring SAML",
             "In Admin > Security > Single sign-on, copy the Orbitly Entity ID and the Assertion Consumer Service URL "
             "into your identity provider. Then upload the identity provider metadata XML or paste the SSO URL and "
             "signing certificate. Click Test connection before enabling SSO for all users.",
             False),
            ("3. Enforcing SSO",
             "When SSO is enforced, users can no longer sign in with a password. The workspace Owner keeps a "
             "break-glass password login in case the identity provider is unavailable.",
             False),
            ("4. SCIM provisioning",
             "SCIM 2.0 provisioning creates, updates and deactivates users automatically from Okta or Microsoft Entra ID. "
             "Generate a SCIM token under Admin > Security > Provisioning. SCIM tokens expire after 1 year and must be rotated.",
             True),
            ("5. Session policy",
             "The default session timeout is 12 hours. Admins can set a value between 15 minutes and 24 hours under "
             "Admin > Security > Session policy. Changing the policy signs out all active sessions.",
             False),
            ("6. Troubleshooting SSO",
             "Error ERR-403 means the SAML response was rejected. The most common causes are an expired signing certificate, "
             "a mismatched Entity ID, or a clock difference of more than 5 minutes between the identity provider and Orbitly.",
             False),
        ],
    ),
    (
        "data_retention_policy.pdf",
        "Data Retention and Deletion Policy",
        "Policy version 2026.1. Effective from 1 January 2026.",
        [
            ("1. Scope",
             "This policy describes how long Orbitly Workspace keeps customer content, audit logs and backups, "
             "and how customers can delete or export their data.",
             False),
            ("2. Deleted items",
             "Deleted files and projects are kept in the workspace trash for 30 days. Owners and Admins can restore "
             "items from the trash during this period. After 30 days items are permanently deleted.",
             False),
            ("3. Audit logs",
             "Audit logs are retained for 1 year on the Standard plan and 7 years on the Enterprise plan.",
             False),
            ("4. Backups",
             "Encrypted backups are kept for 35 days and are used only for disaster recovery. Individual files cannot "
             "be restored from backups on request.",
             True),
            ("5. Workspace deletion",
             "When an Owner deletes a workspace, all content is permanently removed within 30 days, and from backups "
             "within a further 35 days.",
             False),
            ("6. Data export",
             "Owners can export all workspace content as a ZIP archive from Admin > Settings > Export data. "
             "Exports are available for download for 7 days.",
             False),
        ],
    ),
    (
        "support_sla_policy.pdf",
        "Support and Service Level Agreement Policy",
        "Effective from 1 April 2026.",
        [
            ("1. Uptime commitment",
             "Orbitly commits to 99.9% monthly uptime on the Standard plan and 99.95% on the Enterprise plan, "
             "excluding scheduled maintenance.",
             False),
            ("2. Scheduled maintenance",
             "Scheduled maintenance takes place on Sundays between 02:00 and 04:00 UTC and is announced at least "
             "72 hours in advance on the status page.",
             False),
            ("3. Support channels",
             "Standard plan customers get email support during business hours, Monday to Friday. Enterprise customers "
             "get 24/7 email and phone support and a named customer success manager.",
             False),
            ("4. Response times",
             "First response targets for Enterprise: Severity 1 within 1 hour, Severity 2 within 4 hours, Severity 3 "
             "within 1 business day. Standard plan: all severities within 1 business day.",
             True),
            ("5. Service credits",
             "If monthly uptime falls below the commitment, customers may request a service credit within 30 days "
             "of the end of the month. Credits are 10% of the monthly fee for each full 0.1% below the commitment, "
             "up to a maximum of 50%.",
             False),
        ],
    ),
]


def build_pdf(filename: str, title: str, meta: str, sections: list[tuple[str, str, bool]]) -> Path:
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_title(title)
    pdf.set_author("Orbitly Inc. (fictional)")
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, NOTICE, new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(0, 5, meta, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    for heading, body, new_page in sections:
        if new_page:
            pdf.add_page()
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(0, 8, heading, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, body, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)
    out = OUT_DIR / filename
    pdf.output(str(out))
    return out


def main() -> None:
    for filename, title, meta, sections in DOCUMENTS:
        print("wrote", build_pdf(filename, title, meta, sections).name)


if __name__ == "__main__":
    main()
