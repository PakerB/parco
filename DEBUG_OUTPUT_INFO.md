# Debug Output at Max Steps Exceeded

## Summary
When training hits `max_steps` (currently 500), the system will now print comprehensive debugging information to help identify where the environment or model is getting stuck.

## Output Format

When max_steps is exceeded, you'll see:

```
================================================================================
DEBUG INFO AT MAX_STEPS EXCEEDED
================================================================================

Batch index: 0
Number of agents: 4
Current done status: [False, False, False, False]
Step reached: 501

Current node (all agents):
  [5 5 0 2]

Previous action (all agents):
  [5 5 0 2]

Action mask shape: torch.Size([4, 21])
Action mask for first agent (valid actions per node):
  Agent 0: 3 valid nodes -> [0, 5, 20]...
  Agent 1: 2 valid nodes -> [0, 10]...

Detailed constraint analysis for agent 0:
  Visited nodes: 8 out of 21
  Capacity left: 45.30
  Time left: 240.50
  Slack time: 125.30
  Trip deadline reached: False
  Done agents: [False, False, False, False]

Full TensorDict keys available in td
================================================================================
```

## What Each Section Means

### Basic State Info
- **Batch index**: Which batch element (usually 0)
- **Number of agents**: 4 agents (2 trucks + 2 drones)
- **Current done status**: Which agents have finished (all False = stuck)
- **Step reached**: How many steps were taken (501 if max_steps=500)

### Current/Previous Node Status
- **Current node**: Which node each agent is currently at
  - 0 = depot (base)
  - 1-20 = customer nodes
- **Previous action**: Which node each agent was at in previous step
  - If current_node == previous_action → **ACTION REPEATING** (stuck!)

### Action Mask Analysis
- **Shape**: [num_agents, num_nodes] - which nodes are valid actions
- **Agent info**: For each agent shows:
  - How many valid nodes are available
  - Which nodes can be visited next

### Constraint Analysis
- **Visited nodes**: How many customers have been serviced
- **Capacity left**: Remaining load capacity
- **Time left**: Time remaining before route ends
- **Slack time**: Buffering time available
- **Trip deadline reached**: If perishable items expired
- **Done agents**: Which agents are finished

## How to Use This Information

1. **Check if action is repeating**:
   - If `current_node == previous_action` → agent stuck in infinite loop
   - Fix: stuck detection may not be working properly

2. **Check which constraint is blocking all actions**:
   - If valid actions is 0 → constraint logic preventing all nodes
   - Look at capacity/time/deadline to see which is violated

3. **Check if depot is available**:
   - If node 0 (depot) not in valid actions → stuck detection blocking it
   - This might prevent agent from recovering

4. **Compare across agents**:
   - If some agents stuck (low valid actions) and others normal → unequal constraint pressure
   - May indicate constraint calc bug

## When to Stop Training

Once you see this output:
1. **Check the debug info carefully**
2. **Note down which constraints are problematic**
3. **Stop training** (Ctrl+C in terminal)
4. **Share the debug output** so we can identify the root cause

## Files Modified

- `parco/models/policy.py` - Added debug output in forward() method around line 195-262
