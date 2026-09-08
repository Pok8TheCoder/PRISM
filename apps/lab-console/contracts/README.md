# API contracts

Shared between UI and BFF.

| File | Purpose |
|------|---------|
| `openapi.yaml` | OpenAPI 3.1 summary |
| `fixtures/*.json` | Example JSON responses |

TypeScript types in `../frontend/src/types/` are the source of truth for the UI. When the API changes, update types + fixtures + this folder together.

## Fixtures

| File | Endpoint |
|------|----------|
| `dashboard.json` | Partial dashboard snapshot |
| `session.json` | Minimal session metadata |
| `session_full.json` | Full Lab Session chart payload |
| `explain_block.json` | SHAP / Explain tab example |
| `recordings_list.json` | Recordings library example |

## Validate (optional)

```bash
# If you install openapi-cli globally:
npx @redocly/cli lint openapi.yaml
```
