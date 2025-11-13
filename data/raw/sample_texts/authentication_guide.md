# Authentication System Documentation

## Overview

Our authentication system uses JWT tokens with refresh token rotation for enhanced security.

## Architecture

The authentication flow consists of three main components:

1. **Authentication Service** - Handles user login and token generation
2. **Token Validator** - Validates JWT tokens on each request
3. **Session Manager** - Manages active user sessions and token refresh

## JWT Token Structure

```json
{
  "user_id": "uuid-v4",
  "email": "user@example.com",
  "roles": ["user", "admin"],
  "exp": 1234567890,
  "iat": 1234567890
}
```

## Security Features

- Token expiration: 15 minutes for access tokens
- Refresh token rotation on each use
- Rate limiting: 5 login attempts per minute per IP
- Secure password hashing using bcrypt with salt rounds of 12

## Error Codes

- `AUTH_001`: Invalid credentials
- `AUTH_002`: Token expired
- `AUTH_003`: Insufficient permissions
- `AUTH_004`: Rate limit exceeded

## Implementation Notes

The `validateToken()` function must be called on every protected route. Tokens are transmitted via the `Authorization: Bearer <token>` header.
