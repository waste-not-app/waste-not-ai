# Android, Go, and AI Integration

The Android app talks only to the Go API at `https://waste-not-api.bccdev.id/v1/`. The AI workers are private, Gemini is called only by the recommendation worker, and the model detects shapes and classes but never decides the material from a photo.

## Responsibilities

| Layer | Responsibility |
|---|---|
| Android | Single-photo capture, orientation and bounding box preview, confirmation of object, material, and condition, status and source display, operation UUIDs, explicit saves |
| Go | User JWT and sessions, media and schema validation, quotas, ownership, TTL, durable deduplication, provenance, waste log transactions, XP and statistics |
| Detection worker | CPU inference, model checksum, raw classes and boxes, confidence bands and thresholds, candidate mapping |
| Recommendation worker | Guide policy, context and evidence validation, Gemini calls, and exactly one method or a non-`READY` result |

Ports 8091 and 8093 listen on loopback with separate service tokens. Android never receives those tokens or the Gemini keys.

## Paired contract changes

`contracts/v1/public.openapi.json` is the contract the mobile app consumes. The two private OpenAPI documents form the Go–Python contract. The backend keeps a reviewed snapshot in `contracts/ai/`. [`contracts/backend-pin.json`](../contracts/backend-pin.json) here and `contracts/ai.lock.json` in the backend pin the same 34 artifacts.

CI never regenerates contracts to make tests pass. Regenerate them only for an intentional change. Then review the diff, sync the snapshot and pin in both repositories, and check both checkouts together before merging:

```sh
PYTHONPATH=src python -m wastenot_ai.cli export-contracts --check
python scripts/check_contracts.py contracts/v1 contracts/backend-pin.json \
  --peer ../waste-not-backend/contracts/ai
```

A schema change also requires updates to the affected Android DTOs, fixtures, and tests. Fixture examples are simulations and must never become a production success fallback.

## Flow

1. **Load metadata.** Read `metadata` and `ai/objects` through Go with the user's bearer token. Respect the capabilities and guide status they report.
2. **Detect.** Call `POST ai/detections` with a multipart `image` (JPEG or PNG) and a `client_request_id` UUID.
   - There is no base64 upload, no video streaming, and no permanent photo storage.
   - Images are limited to 5 MiB, 12 megapixels, and 4096 px per side.
   - Use the confidence bands and thresholds the server returns.
3. **Confirm.** The user confirms the object key, category, and material, and fills in the context. Send:
   - `input_source`;
   - the detection and object IDs when the object came from a scan;
   - `confirmed_object=true`;
   - `context_revision`.

   Never send prompts, model names, providers, raw classes, or confidence values from the client.
4. **Recommend.** Call `POST ai/recommendations`.
   - `READY` carries exactly one method, scoped to `HANDLING` or `PREVENTION`.
   - Never treat `NEEDS_CONTEXT` or `UNAVAILABLE` as a ready recommendation.
   - Conditional results list the conditions that still need to be verified. Sources and cautions stay visible.
   - Nothing is logged and no XP is awarded at this stage.
5. **Save.** Call `POST waste-logs` only after the user confirms the method and enters a quantity. Handling provenance uses the recommendation ID and both confirmation flags; Reduce (prevention) is logged manually. Discard the session with `DELETE ai/detections/{id}`.

### Idempotency and timeouts

- **One UUID, one operation.** The same UUID always means the same operation. Replay a request with an unknown outcome using its original UUID and payload so data is never duplicated.
- **Replays.** The AI layer replays stored results and errors without calling the provider again. A new generation after a failure requires an explicit user action and a new UUID.
- **Timeout budget.** The deadlines are layered: provider 17 s, worker 18 s, Android read 20 s, then Go 23 s for headers and 25 s in total. Do not extend a single layer or add hidden retries.

## Pilot scope

| Component | Pilot configuration |
|---|---|
| Detection | The v4 waste model with the COCO11s companion (`broad-coco-b0f633193403`): 88 labels, square 640 px input. Both checkpoints must be verified in `AI_MODEL_DIRECTORY` |
| Recommendations | Four source-reviewed v6 guides: PET, HDPE, non-aerosol aluminium cans, and cardboard. They are valid until 4 December 2026 and require confirmed local collector access |

Unit, schema, and HTTP tests do not prove any of the following:

- material accuracy on an independent holdout;
- safety for human handling;
- provider capacity;
- real device camera behavior;
- visual acceptance.

Android must be tested on API 30 and on the latest Android release.
