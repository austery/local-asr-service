---
specId: ADR-003
title: Closed-Source Cloud Diarization as Experimental Baseline
status: 🟡 Proposed
creationDate: 2026-11-11
lastUpdateDate: 2026-11-13
deciders:
  - User (AI-Assisted)
relatedSpecs:
  - ADR-002
tags:
  - project-boundary
  - cloud-baseline
  - diarization
---

# ADR-003: Closed-Source Cloud Diarization as Experimental Baseline

**Date**: 2026-11-11
**Status**: 🟡 Proposed (owner approval pending)
**Deciders**: User (AI-Assisted)

This decision extends the boundary in
[ADR-002](ADR-002-Lightweight-Local-Speech-Gateway-Boundary.md). The text
was first drafted on 2026-11-11 as an in-place amendment to ADR-002. It moved
here on 2026-11-13, because a changed decision gets its own ADR.

## Context

The owner wants to evaluate local ASR and diarization quality against
commercial closed-source baselines, using vendor credits.

## Decision

The gateway boundary admits explicit, thin file-transcription adapters for
selected cloud providers (for example Deepgram Nova-3, OpenAI).

Invariants governing cloud adapters:

1. **Experimental baseline purpose**: Cloud adapters exist strictly for
   comparative quality evaluation and benchmarking. They do not alter the
   gateway's core identity as a local-first speech gateway.
2. **Strict opt-in**: Requests must explicitly specify the cloud alias (for
   example `deepgram-nova-3-diarize`). No automatic fallback from local to
   cloud is permitted.
3. **Thin adapter boundary**: Each provider performs transcription and
   diarization together via upstream APIs. This repository owns only the thin
   HTTP adapter, credential gating, and response normalization. Local ASR text
   is never combined with cloud diarization turns.
4. **Bypass resident workers**: Cloud HTTP calls run asynchronously. They do
   not hold the resident model worker lock or spawn local worker processes.

## Alternatives

Not recorded.

## Consequences

- A request routed to a cloud alias is a paid call and sends audio to a third
  party. The impact radius map records this as a One-way effect.
- The live acceptance of the Deepgram adapter is tracked separately (Issue #62).
