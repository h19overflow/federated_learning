# Daedalus Orchestrator Design Document

## Overview
**Daedalus** is the master builder orchestrator agent that coordinates requests across specialized agents (Athena, Atlas, Icarus, Hermes). Named after the Greek mythological master craftsman who designed the Labyrinth and Icarus's wings, Daedalus embodies methodical planning, careful design, and collaborative execution.

## Core Philosophy
- **Think Before Building**: All agents produce plan files before execution
- **User-Driven Approval**: Plans are presented for user review and discussion
- **Collaborative Design**: Users can debate plans with Daedalus before proceeding
- **Builder Energy**: Focus on construction, architecture, and systematic thinking

## Architecture

### Agent Hierarchy
```
┌─────────────────────────────────────────┐
│         DAEDALUS (Orchestrator)         │
│    Debate Intent → Route → Wait → Approve
└─────────────────────────────────────────┘
         ↓         ↓         ↓         ↓
    ┌────────┬────────┬────────┬────────┐
    │ ATHENA │ ATLAS  │ ICARUS │ HERMES │
    │Strategy│Execute │Complex │ Speed  │
    │Planning│ Heavy  │Orchest │Execution
    │        │ Lift   │ration  │        │
    └────────┴────────┴────────┴────────┘
```

### Agent Responsibilities

| Agent | Role | When to Use | Output |
|-------|------|------------|--------|
| **Athena** | Strategic Planning & Design | Complex architecture, design decisions, multi-step strategy | `.md` plan file with approach, trade-offs, reasoning |
| **Atlas** | Execution & Infrastructure | Heavy lifting, infrastructure setup, resource allocation | `.md` plan file + execution logs |
| **Icarus** | Complex Orchestration | Multi-agent coordination, intricate workflows, parallel delegation | `.md` plan file with coordination strategy |
| **Hermes** | Direct Execution & Speed | Quick fixes, simple changes, fast execution | `.md` plan file + immediate execution |

## Request Flow

### Step 1: Intent Clarification & Debate
```
User Request
    ↓
Daedalus analyzes intent
    ↓
Daedalus debates/questions with user
    ↓
User confirms intent
```

### Step 2: Route to Specialists
```
Daedalus determines complexity & domain
    ↓
Routes to appropriate agent(s):
  - Simple task → Hermes
  - Heavy infrastructure → Atlas
  - Complex coordination → Icarus
  - Strategic design → Athena
  - Mixed → Multiple agents
```

### Step 3: Planning Phase
```
Each selected agent writes:
  `.claude/planning/[task-name]-[agent].md`
    
Plan includes:
  - Approach & reasoning
  - Trade-offs & alternatives considered
  - Step-by-step execution plan
  - Risk assessment
  - Validation criteria
```

### Step 4: Approval Gate
```
Daedalus presents all plans to user
    ↓
User reviews & approves OR requests changes
    ↓
User can discuss plans with Daedalus
    ↓
Once approved → proceed to execution
```

### Step 5: Execution
```
Agents execute according to approved plans
    ↓
Daedalus monitors & coordinates
    ↓
Results reported back to user
```

## Plan File Specification

### Location
```
.claude/planning/
├── [task-name]-athena.md
├── [task-name]-atlas.md
├── [task-name]-icarus.md
└── [task-name]-hermes.md
```

### Content Structure
```markdown
# [Task Name] - [Agent Name] Plan

## Summary
[1-2 sentence overview]

## Approach
[Detailed methodology]

## Trade-offs
- Option A: [pros/cons]
- Option B: [pros/cons]
- Selected: [Why this option]

## Execution Steps
1. [Step 1]
2. [Step 2]
...

## Risk Assessment
- Risk 1: [mitigation]
- Risk 2: [mitigation]

## Validation Criteria
- [ ] Criterion 1
- [ ] Criterion 2

## Dependencies
- [External dependencies or prerequisites]
```

## Key Design Decisions

### 1. All Agents Write Plans First
**Why**: Ensures user has visibility and approval before any action
**Benefit**: Prevents wasted work, catches issues early
**Trade-off**: Slightly slower initial response, but higher quality outcomes

### 2. Daedalus Waits for Approval
**Why**: User is the decision-maker, not the system
**Benefit**: Full user control, collaborative design
**Trade-off**: Requires user engagement

### 3. Users Can Discuss Plans with Daedalus
**Why**: Plans may need refinement based on user feedback
**Benefit**: Iterative improvement, better final outcomes
**Trade-off**: Multi-turn conversation required

### 4. Daedalus Routes to Multiple Agents if Needed
**Why**: Complex tasks often require multiple specialists
**Benefit**: Comprehensive coverage, parallel planning
**Trade-off**: Coordination overhead

## Interaction Patterns

### Pattern 1: Simple Task (Hermes)
```
User: "Add a button to the dashboard"
  ↓
Daedalus: "Quick fix, right? Button placement, styling, integration?"
  ↓
User: "Top right, blue, links to settings"
  ↓
Hermes writes plan
  ↓
User approves
  ↓
Hermes executes
```

### Pattern 2: Complex Task (Multiple Agents)
```
User: "Redesign the authentication system"
  ↓
Daedalus: "Big change. Security implications? Migration strategy? Timeline?"
  ↓
User: "Yes, all critical"
  ↓
Athena + Atlas write plans
  ↓
User reviews, discusses with Daedalus
  ↓
Plans refined
  ↓
User approves
  ↓
Athena + Atlas execute
```

### Pattern 3: Plan Discussion
```
User: "I don't like this approach"
  ↓
Daedalus: "What's the concern? Let's explore alternatives"
  ↓
Daedalus + User debate
  ↓
Plans revised
  ↓
User approves new plan
  ↓
Execution proceeds
```

## Implementation Checklist

- [ ] Create Daedalus agent definition
- [ ] Create Daedalus command (`.claude/commands/daedalus`)
- [ ] Modify Athena to output plan files
- [ ] Modify Atlas to output plan files
- [ ] Modify Icarus to output plan files
- [ ] Modify Hermes to output plan files
- [ ] Implement approval gate mechanism
- [ ] Create plan file templates
- [ ] Document agent routing logic
- [ ] Test multi-agent coordination

## Success Criteria

✅ User can request any task and Daedalus debates intent
✅ All agents produce plan files before execution
✅ User can review and approve all plans
✅ User can discuss plans with Daedalus
✅ Plans are executed only after explicit approval
✅ Multi-agent tasks are coordinated seamlessly
✅ Plan files are organized and discoverable

## Future Enhancements

- Plan versioning (track iterations)
- Plan comparison (show alternatives side-by-side)
- Automated plan validation
- Plan templates by task type
- Integration with git for plan history
