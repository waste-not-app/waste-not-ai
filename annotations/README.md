# Research annotation provenance

`realwaste-cardboard-v1.json` records 60 review decisions and 15 manual cardboard bounding boxes by Codex (7 November 2026). Source: Sam Single et al., [RealWaste](https://github.com/sam-single/realwaste/tree/156e8e9154eeb163df30fcfd767f55b19633bb06), commit `156e8e9154eeb163df30fcfd767f55b19633bb06`. These derived annotations follow **CC BY-NC-SA 4.0**, as stated by the author. Changes: manual pixel XYXY boxes, image hashes and review decisions; independent human review remains pending. UCI lists a different license; this batch uses the stricter author terms and is restricted to noncommercial research. [License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

`fpi-det-sample-v1.json` records phone-scope screening of 60 publisher images by Codex. Source: [FPI-Det](https://github.com/KvCgRv/FPI-Det/tree/83c452d1a1e5b1bac88afe1a6834cb9c25e036eb), commit `83c452d1a1e5b1bac88afe1a6834cb9c25e036eb`, whose publisher states MIT licensing. Changes: hashes, review status and view tags; phone/face label scope is retained. This is audit metadata and does not certify complete COCO annotations or independent sessions.

Raw source photographs, archives, local datasets and trained weights are excluded from Git. See [the audit and limitations](../docs/archive/EXTERNAL_DATASET_AUDIT_V5.md).
