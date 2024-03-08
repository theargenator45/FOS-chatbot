REACT = """
You also have access to actions, which you should use whenever appropriate in order to complete tasks.

ACTIONS:
------

{tools}

To use a action, please use the following format:

```
Thought: Do I need to use a action? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the user, or if you do not need to use a action, you MUST use the format:

```
Thought: Do I need to use a action? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
"""