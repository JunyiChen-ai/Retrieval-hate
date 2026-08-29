# LB-SCGP G0 Round2 Fix Handoff

This file mirrors `G0_IMPLEMENTATION_HANDOFF.md` as the explicit Round2 repair handoff requested for the next independent review.

**Date:** 2026-07-11  
**Launcher metadata:** model `gpt-5.5`, `model_reasoning_effort=xhigh`, `--strict-config`.  
**Execution:** no SLURM job, no login-node Python experiment computation, no sanitizer/G0 artifact, no performance result, no teacher/MLLM/OCR call.

## Review Scope

Review the Round2 repair against the five CRITICAL and two HIGH findings in `G0_INDEPENDENT_REVIEW_ROUND1.md`. Do not treat this handoff as a 0C/0H self-certification.

## Required Re-Review Questions

- Is the quarantine sanitizer exception principled, transparent, and still leakage-safe?
- Does formal G0 avoid mixed-cache locators/hashes and mixed/protected path hashing?
- Can the real Dykstra/rank-cell gate be independently replayed from emitted evidence?
- Does the separate GPU replay prove actual checkpoint fit and live-state rollback?
- Does the registered cone cover singleton/pair/triplet/SupCon families with a genuine separation oracle?
- Is H10 exactly the registered formula with final bank outside the refresh term?
- Are one-GPU resources verified from SLURM and torch runtime, not self-report?

## Current Decision

Prepared for independent review only. STOP for execution until that review passes and the user authorizes sanitizer/G0 SLURM jobs.
