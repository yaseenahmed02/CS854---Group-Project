# Production Deployment Checklist

## Pre-Deployment

- [ ] All tests passing (unit, integration, e2e)
- [ ] Code review completed and approved
- [ ] Database migrations tested on staging
- [ ] Performance benchmarks meet SLA requirements
- [ ] Security scan completed (OWASP Top 10)
- [ ] Dependency vulnerabilities resolved

## Infrastructure

- [ ] Auto-scaling configured (min: 3, max: 20 instances)
- [ ] Health check endpoints responding correctly
- [ ] Load balancer configured with proper health checks
- [ ] CDN cache invalidation strategy in place
- [ ] Monitoring dashboards created in Grafana
- [ ] Alert rules configured in PagerDuty

## Database

- [ ] Backup strategy verified (hourly snapshots, 30-day retention)
- [ ] Connection pool sized appropriately (max: 100 connections)
- [ ] Indexes created for new queries
- [ ] Query performance analyzed with EXPLAIN

## Security

- [ ] Environment variables set (no hardcoded secrets)
- [ ] SSL certificates valid and auto-renewal configured
- [ ] CORS policies reviewed
- [ ] Rate limiting enabled (1000 req/min per user)
- [ ] API keys rotated

## Rollback Plan

1. Keep previous 3 versions in container registry
2. Database migrations must be backward-compatible
3. Feature flags enable instant rollback without redeployment
4. DNS TTL set to 60 seconds for quick switchover

## Post-Deployment Verification

- [ ] Smoke tests passed
- [ ] Error rate <0.1%
- [ ] P95 latency <500ms
- [ ] No memory leaks detected in first hour
- [ ] Log aggregation working correctly
