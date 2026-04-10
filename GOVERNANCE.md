# MYCELIUM — Network Governance Model

> *"Governance by the network, for the network, of the network."*

This document describes how decisions are made in the Mycelium project and network, both now and as it scales.

---

## Table of Contents

1. [Governance Philosophy](#1-governance-philosophy)
2. [Current Governance (v0.2.0)](#2-current-governance-v020)
3. [Transitional Governance (v0.5-v1.0)](#3-transitional-governance-v05-v10)
4. [Mature Governance (v2.0+)](#4-mature-governance-v20)
5. [Decision-Making Processes](#5-decision-making-processes)
6. [Roles & Responsibilities](#6-roles--responsibilities)
7. [Proposal System](#7-proposal-system)
8. [Voting Mechanisms](#8-voting-mechanisms)
9. [Conflict Resolution](#9-conflict-resolution)
10. [Anti-Capture Mechanisms](#10-anti-capture-mechanisms)
11. [Governance Metrics](#11-governance-metrics)
12. [Sovereign Clusters](#12-sovereign-clusters)

---

## 1. Governance Philosophy

### 1.1 Principles

1. **Minimal governance** — Govern only when necessary; maximize autonomy
2. **Transparency** — All decisions and processes are visible
3. **Meritocratic weighting** — Those who contribute more have more influence
4. **Progressive decentralization** — Start centralized, decentralize over time
5. **Reversibility** — Decisions can be undone if they prove harmful

### 1.2 What Governance Covers

| Domain | Examples | Governance Mechanism |
|--------|----------|---------------------|
| **Protocol** | Message formats, P2P rules | RFC process + community vote |
| **Model weights** | Trusted hash registry | DHT consensus |
| **Code** | Merging PRs, releases | Maintainer review |
| **Ethics** | Usage policies, restrictions | Community discussion |
| **Resources** | Bootstrap nodes, funding | Operator discretion |

### 1.3 What Governance Does NOT Cover

- **Individual node behavior** — Nodes are autonomous
- **User data** — Users control their own data
- **Local models** — Users choose what to run
- **Personal opinions** — Contributors speak for themselves

---

## 2. Current Governance (v0.2.0)

### 2.1 Structure: Benevolent Dictator

**Current state**: Single maintainer makes final decisions

```
┌─────────────┐
│  Maintainer │
│  (decision) │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Community  │
│  (input)    │
└─────────────┘
```

**How it works**:
- Maintainer has final say on all decisions
- Community provides input via issues and PRs
- Maintainer solicits feedback on significant changes
- Decisions are documented and explained

### 2.2 Why This Works Now

- **Small project** — One person can understand the whole system
- **Fast decisions** — No bureaucracy, rapid iteration
- **Clear accountability** — One person is responsible
- **Community trust** — Maintainer has earned trust through work

### 2.3 Limitations

- **Bus factor** — Project depends on one person
- **Bottleneck** — Maintainer capacity limits progress
- **Single perspective** — Limited viewpoint
- **Scaling** — Won't work at 1,000+ contributors

---

## 3. Transitional Governance (v0.5-v1.0)

### 3.1 Structure: Core Team + RFC Process

**Trigger for transition**: 5+ active contributors, 100+ stars

```
┌─────────────────────────────────────┐
│          Core Team (5-9)            │
│  ┌───────────┬───────────┬────────┐ │
│  │Technical  │Community  │Security│ │
│  │Lead       │Lead       │Lead    │ │
│  └───────────┴───────────┴────────┘ │
└────────────────┬────────────────────┘
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
   ┌────────┐ ┌────────┐ ┌────────┐
   │  WG 1  │ │  WG 2  │ │  WG N  │
   │Arch    │ │Security│ │Perf    │
   └────────┘ └────────┘ └────────┘
```

### 3.2 Core Team

**Composition**: 5-9 active contributors

**Selection**:
- Current maintainer invites contributors
- Based on sustained contribution (6+ months)
- Core team approves by 2/3 majority
- Term: Ongoing, can be removed by 2/3 vote

**Responsibilities**:
- Review and merge significant PRs
- Propose and vote on RFCs
- Set project direction
- Manage releases

### 3.3 Working Groups

**Structure**: Focused teams for specific areas

| Working Group | Focus | Autonomy |
|--------------|-------|----------|
| **Architecture** | Protocol design, crate structure | High (core team oversight) |
| **Security** | Audits, threat modeling, vuln response | High (can act independently in emergencies) |
| **Performance** | Benchmarks, optimization | Medium (coordinates with Architecture) |
| **Community** | Documentation, onboarding, events | High |
| **Research** | Algorithm improvements, papers | High |

### 3.4 RFC Process

**For significant changes** (protocol changes, major features):

```
1. Draft RFC → 2. Community Comment (2 weeks) → 3. Core Team Vote → 4. Implementation
```

**RFC Template**:
```markdown
# RFC-XXX: Title

## Summary
Brief description of the change

## Motivation
Why this is needed

## Design
Detailed technical design

## Alternatives Considered
Other approaches and why they weren't chosen

## Implementation Plan
How this will be implemented

## Voting
- [ ] Core Team Member 1
- [ ] Core Team Member 2
- ...
```

**Approval**: 2/3 majority of core team

---

## 4. Mature Governance (v2.0+)

### 4.1 Structure: Elected Steering Committee

**Trigger**: 20+ active contributors, 1,000+ nodes

```
┌─────────────────────────────────────────┐
│      Community (all contributors)       │
│              │                          │
│              ▼ (election, annual)       │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │   Steering Committee (7-11)     │   │
│  │  ┌─────────┬─────────┬────────┐ │   │
│  │  │ Chair   │ Sec     │ Treas  │ │   │
│  │  └─────────┴─────────┴────────┘ │   │
│  └──────────────┬──────────────────┘   │
└─────────────────┼──────────────────────┘
                  │ delegates to
         ┌────────┼────────┐
         ▼        ▼        ▼
    ┌────────┐ ┌────────┐ ┌────────┐
    │  WG 1  │ │  WG 2  │ │  WG N  │
    └────────┘ └────────┘ └────────┘
```

### 4.2 Steering Committee

**Election**:
- Annual election by all contributors
- Contributors defined as: 10+ merged PRs or 6+ months active participation
- Seats: 7-11 (odd number to avoid ties)
- Term: 2 years, staggered (half elected each year)

**Powers**:
- Approve RFCs
- Set project priorities
- Allocate resources (grants, sponsorships)
- Represent project externally
- Amend governance document (requires 3/4 majority)

**Limits**:
- Cannot change license without 90% community approval
- Cannot merge code without working group review
- Cannot make unilateral technical decisions
- Subject to recall by 2/3 community vote

### 4.3 Network-Level Governance

**For the running network** (not just the project):

```
┌─────────────────────────────────────────┐
│         All Network Nodes               │
│              │                          │
│              ▼ (weight by contribution) │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │     Protocol Proposals          │   │
│  │   (voting by node operators)    │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

**Voting weight**: Based on compute contribution (see §8.2)

**Proposable**:
- Protocol parameter changes
- Weight registry updates
- Network-wide policy changes
- Emergency interventions

---

## 5. Decision-Making Processes

### 5.1 Decision Types

| Type | Scope | Process | Examples |
|------|-------|---------|----------|
| **Routine** | Day-to-day | Individual discretion | Bug fixes, docs updates |
| **Significant** | Feature-level | RFC + core team vote | New protocol feature |
| **Major** | Project-level | Community vote | License change, fork |
| **Emergency** | Security crisis | Security WG + rapid response | Vulnerability patch |

### 5.2 Decision Matrix

```
                    Impact
                 Low    High
             ┌────────┬────────┐
         Low │ Routine│Signif. │
Reversibility│        │        │
             ├────────┼────────┤
        High │Signif. │ Major  │
             │        │        │
             └────────┴────────┘
```

### 5.3 Consensus-Seeking

**Before voting, seek consensus**:

1. **Present proposal** clearly
2. **Listen to concerns** actively
3. **Incorporate feedback** where possible
4. **Address objections** respectfully
5. **Call vote** only when further discussion is unproductive

**Consensus doesn't mean unanimity** — it means everyone feels heard and can accept the decision.

---

## 6. Roles & Responsibilities

### 6.1 Role Definitions

| Role | Definition | Rights | Responsibilities |
|------|-----------|--------|-----------------|
| **User** | Runs a node | Use the network | Follow protocol rules |
| **Contributor** | 10+ merged PRs | Vote in elections | Maintain code quality |
| **Maintainer** | Core team member | Merge PRs, propose RFCs | Guide project direction |
| **WG Lead** | Working group coordinator | Set WG agenda | Facilitate WG work |
| **Committee Member** | Elected representative | Steering committee vote | Represent community |

### 6.2 Role Progression

```
User → Contributor → Maintainer → Committee Member
  │         │            │              │
  │         │            │              └─ Elected by community
  │         │            └─ Invited by core team
  │         └─ 10+ merged PRs
  └─ Runs a node
```

### 6.3 Term Limits & Rotation

| Role | Term | Limit |
|------|------|-------|
| Core Team Member | Ongoing | Can be removed by 2/3 vote |
| WG Lead | 1 year | 2 consecutive terms max |
| Committee Member | 2 years | 2 consecutive terms max |

---

## 7. Proposal System

### 7.1 Proposal Types

| Type | Author | Reviewers | Approval |
|------|--------|-----------|----------|
| **RFC** | Anyone | Core team | 2/3 core team |
| **Community Proposal** | Contributor | Community | Simple majority |
| **Emergency Proposal** | Anyone | Security WG | WG lead + 2 members |

### 7.2 Proposal Lifecycle

```
Draft → Review → Revision → Vote → Implementation → Review
 │        │         │         │          │            │
 └─ Author └─ Community └─ Address  └─ Decision └─ Merge  └─ Post-
    writes    comments   concerns             code     implement
                                                       review
```

### 7.3 Proposal Requirements

All proposals must include:
1. **Problem statement** — What problem does this solve?
2. **Proposed solution** — How does it work?
3. **Alternatives considered** — Why this approach?
4. **Impact assessment** — Who/what is affected?
5. **Implementation plan** — How will it be done?
6. **Risks and mitigations** — What could go wrong?

---

## 8. Voting Mechanisms

### 8.1 Project Voting (GitHub)

**Who votes**: Contributors (10+ merged PRs or 6+ months active)

**Methods**:
- **Simple majority** — >50% approval (routine decisions)
- **Supermajority** — 2/3 approval (significant decisions)
- **Consensus** — No strong objections (minor decisions)

**Tools**: GitHub discussions, polls, or dedicated voting platform

### 8.2 Network Voting (On-Chain)

**Who votes**: Node operators

**Voting weight**: Based on compute contribution

```rust
pub struct VotingWeight {
    node_id: NodeId,
    compute_hours: u64,          // Total compute contributed
    gradients_shared: u64,       // Learning contributions
    uptime_hours: u64,           // Network stability
    reputation: f64,             // 0.0 - 1.0
}

impl VotingWeight {
    pub fn score(&self) -> f64 {
        let compute_score = (self.compute_hours as f64).log10();
        let gradient_score = (self.gradients_shared as f64).log10();
        let uptime_score = (self.uptime_hours as f64).log10();

        (compute_score * 0.4
         + gradient_score * 0.3
         + uptime_score * 0.2
         + self.reputation * 10.0 * 0.1)
    }
}
```

**Why logarithmic?** Prevents whale dominance — 1000x compute ≠ 1000x votes

### 8.3 Quorum Requirements

| Decision Type | Quorum | Approval |
|--------------|--------|----------|
| Routine | N/A | Simple majority of voters |
| Significant | 25% of contributors | 2/3 of voters |
| Major | 50% of contributors | 3/4 of voters |
| Emergency | N/A | Security WG lead + 2 members |

---

## 9. Conflict Resolution

### 9.1 Escalation Path

```
1. Direct discussion between parties
        ↓ (if unresolved)
2. Mediation by neutral third party
        ↓ (if unresolved)
3. Working group review
        ↓ (if unresolved)
4. Core team / Steering committee decision
        ↓ (if unresolved)
5. Community vote
```

### 9.2 Mediation

**Mediators**: Respected community members agreed upon by both parties

**Process**:
1. Each party presents their perspective
2. Mediator identifies common ground
3. Mediator proposes compromise
4. Parties accept or escalate

### 9.3 Removal Process

**For removing contributors or maintainers**:

```
1. Document the issue (behavior, impact)
2. Discuss with the individual privately
3. If unresolved, propose removal to core team
4. Core team votes (2/3 required)
5. If approved, remove and document reasons
6. Individual can appeal to community vote
```

**Grounds for removal**:
- Repeated code of conduct violations
- Malicious contributions (backdoors, data theft)
- Persistent low-quality contributions despite feedback
- Community harm (harassment, discrimination)

---

## 10. Anti-Capture Mechanisms

### 10.1 What is Capture?

**Capture**: When a subset of participants gains disproportionate control, undermining the project's mission.

**Forms of capture**:
- **Corporate capture** — Company gains control via resource dominance
- **Ideological capture** — Single viewpoint suppresses diversity
- **Technical capture** — Only a few understand critical systems
- **Social capture** — Personal relationships override merit

### 10.2 Prevention Mechanisms

| Mechanism | Prevents | How |
|-----------|----------|-----|
| **Term limits** | Entrenched power | Regular rotation of leadership |
| **Transparency** | Hidden agendas | All decisions documented |
| **Merit weighting** | Resource dominance | Logarithmic voting prevents whales |
| **Working groups** | Technical silos | Knowledge distributed across WGs |
| **Recall votes** | Unaccountable leaders | Community can remove anyone |
| **Forkability** | Institutional capture | Anyone can fork if governance fails |

### 10.3 Detection Metrics

| Metric | Warning Sign | Threshold |
|--------|-------------|-----------|
| **Decision diversity** | Same people always win votes | <5 unique proposers/month |
| **Contributor churn** | Contributors leaving | >20% departure rate/quarter |
| **Core team homogeneity** | Similar backgrounds | <3 different organizations |
| **Decision speed** | Decisions take too long | >3 months for routine RFCs |
| **Community sentiment** | Dissatisfaction | <60% approval in surveys |

---

## 11. Governance Metrics

### 11.1 Health Indicators

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Contributor growth** | +10%/quarter | GitHub contributors |
| **Decision throughput** | 5+ RFCs/quarter | RFC tracking |
| **Community satisfaction** | >80% approval | Quarterly surveys |
| **Leadership diversity** | 3+ organizations | Core team backgrounds |
| **Conflict resolution time** | <2 weeks | Issue tracking |

### 11.2 Quarterly Review

**Every quarter, assess**:

1. **Is governance working?** — Are decisions made effectively?
2. **Is the community healthy?** — Are contributors satisfied?
3. **Are we on mission?** — Are we serving all sentient beings?
4. **What needs to change?** — Adjust governance as needed

---

## 12. Sovereign Clusters

### 12.1 Concept

**Sovereign clusters** are semi-autonomous sub-networks with their own governance, connected to the broader Mycelium network.

**Use cases**:
- **Enterprise** — Company-specific cluster with internal governance
- **Regulatory** — EU cluster with GDPR-compliant governance
- **Research** — Academic cluster with open science governance
- **Cultural** — Language-specific cluster with local governance

### 12.2 Structure

```
┌─────────────────────────────────────────┐
│           Global Mycelium               │
│                                         │
│  ┌──────────┐  ┌──────────┐  ┌───────┐ │
│  │  EU      │  │  Corp X  │  │  Res. │ │
│  │ Cluster  │◄─┤ Cluster  │◄─┤Cluster│ │
│  │ (GDPR)   │  │(Internal)│  │(Open) │ │
│  └──────────┘  └──────────┘  └───────┘ │
│       │              │            │     │
│       └──────────────┼────────────┘     │
│                      │                   │
│              Global coordination         │
│              (weight registry,           │
│               protocol updates)          │
└─────────────────────────────────────────┘
```

### 12.3 Governance Autonomy

**Sovereign clusters can**:
- Set their own internal governance rules
- Enforce additional privacy requirements
- Control their own weight registries
- Manage their own funding

**Sovereign clusters must**:
- Follow base protocol (P2P rules, message formats)
- Accept global weight registry updates
- Participate in federated learning
- Respect the project's ethical principles

---

## Conclusion

Mycelium's governance model evolves with the project:

1. **Now**: Benevolent dictator — fast, clear, accountable through work
2. **Soon**: Core team + RFCs — distributed expertise, structured decisions
3. **Later**: Elected steering committee — democratic, representative, accountable

**The goal**: Minimal governance that protects the project's mission while maximizing contributor autonomy and community benefit.

---

*This document will evolve as the project grows. All contributors are welcome to propose changes through the RFC process.*

**Last Updated**: April 10, 2026
**Version**: v0.2.0
**Status**: Living document — will be updated as governance matures

---

## See Also

- [PHILOSOPHY.md](PHILOSOPHY.md) — Ethical framework and vision
- [SCALING.md](SCALING.md) — Scaling analysis and strategies
- [CONTRIBUTING.md](CONTRIBUTING.md) — How to contribute
- [SECURITY.md](SECURITY.md) — Security model and threat analysis
