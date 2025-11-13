# Caching Strategy

## Redis Cache Architecture

Our application uses Redis for distributed caching with the following hierarchy:

### Cache Layers

1. **L1 Cache**: In-memory LRU cache (100MB limit)
2. **L2 Cache**: Redis cluster (10GB total)
3. **L3 Cache**: Database query results

### Cache Key Patterns

- User profiles: `user:profile:{user_id}`
- API responses: `api:response:{endpoint}:{hash}`
- Session data: `session:{session_id}`

### TTL Configuration

| Data Type | TTL | Reasoning |
|-----------|-----|-----------|
| User profiles | 1 hour | Rarely changes, frequently accessed |
| API responses | 5 minutes | Dynamic data, needs freshness |
| Static assets | 24 hours | Immutable content |

## Cache Invalidation

We use cache-aside pattern with manual invalidation on writes:

```python
def update_user_profile(user_id, data):
    # Update database
    db.update(user_id, data)
    # Invalidate cache
    cache.delete(f"user:profile:{user_id}")
```

## Performance Metrics

- Cache hit ratio target: >95%
- Average cache latency: <2ms
- Redis cluster max memory policy: allkeys-lru

## Common Issues

**Problem**: Cache stampede during cache miss
**Solution**: Use lock-based approach with setnx command

**Problem**: Stale data after updates
**Solution**: Implement pub/sub based cache invalidation
