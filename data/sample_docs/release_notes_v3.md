# Orbitly Workspace Release Notes (v3.0)

> SYNTHETIC sample document created by the team for development and testing.
> Orbitly Inc. is a fictional company. Released 1 October 2026. Applies to version 3.0.

## Highlights

Version 3.0 introduces SCIM user provisioning, a redesigned admin console and higher API limits.

## Changed limits

The API rate limit for Standard plan accounts is increased from 300 to 500 requests per minute.
The Enterprise plan rate limit remains 1,000 requests per minute.
The default session timeout is extended from 8 hours to 12 hours.
The maximum file upload size is increased from 100 MB to 250 MB.

## New features

SCIM 2.0 provisioning is available for Okta and Microsoft Entra ID (formerly Azure AD).
Workspace owners can export audit logs as CSV from Admin > Audit log > Export.
Webhooks now support custom HTTP headers.

## Deprecations

The legacy v1 REST API will be switched off on 31 March 2027. Migrate to the v2 API before that date.
Password-only sign-in for Enterprise workspaces is deprecated; enforce SSO or MFA instead.

## Known issues

Desktop app 3.0.0 on Windows may fail to sync folders whose path is longer than 260 characters.
Workaround: enable long paths in Windows or move the folder closer to the drive root. Fixed in 3.0.1.
