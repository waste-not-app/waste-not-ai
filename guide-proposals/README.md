# Guide Proposals

[`pilot-guides.draft.json`](pilot-guides.draft.json) contains the four v6 candidate guides. They remain `DRAFT`, have empty reviewer and date metadata, and are never installed on a production worker. The published versions live in [`guides/`](../guides/README.md).

## Candidates

| Candidate | Recycling conditions | Proposed change |
|---|---|---|
| PET bottle | Clean, intact, former food or drink contents, collector confirmed | Former contents are required. Not a guide for chemical bottles or a refill-safety guarantee |
| HDPE bottle | Clean, intact, former food or drink contents, collector confirmed | Keep the contents and resin-code limits. No refill-safety guarantee |
| Aluminium can | Clean, intact, former food or drink contents, collector confirmed | Limited to non-aerosol, unpressurized cans. Chemical exclusions and a warning against puncturing |
| Cardboard | Clean, intact, collector confirmed | Dryness and grease rules follow the collector. No single region's rule becomes universal |

Each candidate has one `RECYCLE` evidence item and one `REDUCE` evidence item.

- **Reduce.** `REDUCE` applies only to `PREVENTION` (the next purchase or use) and never replaces handling of the current item.
- **No inference from photos.** Material, condition, and collector acceptance are never inferred from a photo.

Sources:

- [EPA, common recyclables](https://www.epa.gov/recycle/how-do-i-recycle-common-recyclables)
- [EPA, household hazardous waste](https://www.epa.gov/hw/household-hazardous-waste-hhw)
- [BPOM, plastic resin codes](https://tasikmalaya.pom.go.id/berita/cerdas-memilih-kemasan-wujudkan-pangan-aman-melalui-pengenalan-kode-plastik)

## Remaining work

The other 74 draft guides still need correction and review. The highest-priority groups are:

- plastic cups and bowls;
- vases and broken glass;
- electronics and batteries;
- knives and scissors;
- guides that apply foreign collection routes.

### Before publishing

Before any candidate is published, the owner must:

- review the final text;
- choose the regional scope;
- record a real reviewer identity, review date, and expiry.

For cans, the catalog label and pilot confirmation read "non-aerosol aluminium food or drink can", because the condition schema has no aerosol field. Android must also be able to report former contents.

Never mark every draft as published, record a false reviewer or date, or use demo flags in production.

## Tests

The tests cover the schema and the condition gates:

- the production worker rejects `DRAFT` guides;
- unknown former contents ask for more context;
- collector access `NO` or chemical contamination never yields `RECYCLE`;
- matching input passes through the private prototype path.

They do not replace content review, evidence that a facility accepts the material, Gemini quality checks, or Android testing.
