# Sample Images

This directory contains synthetic images for the multimodal corpus.

## Image Descriptions

### system_architecture.png
A system architecture diagram showing:
- Client Application
- API Gateway
- Authentication Service
- Redis Cache Layer
- PostgreSQL Database
- Arrows showing request flow between components

### dashboard_mockup.png
A dashboard UI mockup showing:
- Header with "Dashboard - User Analytics" title
- Left navigation menu (Home, Analytics, Users, Settings, Logout)
- Main content area with cards:
  - Active Users: 1,234 (+12.5% growth)
  - Response Time (p95): 456ms
  - Cache Hit Ratio: 94.2%
- Recent activity table with user actions

## Generating Images

To generate the actual PNG files, run:
```bash
python3 _generate_images.py
```

Note: Requires Pillow library (`pip install Pillow`)

For the baseline evaluation, these image descriptions will be embedded as text until actual images are generated.
