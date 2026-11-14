# API Usage Limits (v2.0)

> SYNTHETIC sample document created by the team for development and testing.
> Effective from 1 July 2026. Applies to API version 2.0.

This page describes request quotas and throttling for the v2 API.

## Rate limits

Standard plan accounts may send up to 300 requests per minute.
Enterprise plan accounts may send up to 1,000 requests per minute.
Requests above the limit receive HTTP 429 Too Many Requests with a Retry-After header.

## Monthly quotas

Standard plan: 1,000,000 API calls per month.
Enterprise plan: unlimited API calls, subject to fair use.

## Increasing your limit

Workspace administrators can request a temporary limit increase from Settings > Usage > Request increase.
Requests are reviewed within 2 business days.
