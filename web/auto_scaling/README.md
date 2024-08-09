# Automatic Scaling

## Motivation

Streamline developer workflow by automatically:

- Bring up model instance when requested but none is running.
- Stop model instances when no more requests are being made

Additionally:

- Automatically spin up additional instances (scale-up) when needed
  - E.g., when concurrent requests for current instance exceeds some threshold.

## Background

KScope provides the following functionalities:

- Ability to bring specific model instances up and down
- Global view of the status of all model instances

Additionally, the KScope OpenAI compatibility layer implements the following:

- Ability to trigger function calls when a request arrives
- Visibility into the number of token returned in each request
- Visibility into whether each request is streaming or not.

## Proposed Workflow

The proposed setup consists of two components:

- An auto-scaling thread for launching and stopping LLM worker instances
- Web server threads (Flask-managed) for updating usage stats for each LLM.

The following should be specified in a configuration file:

- Maximum number of replicas permitted for each LLM
- Minimum number of replicas for each LLM
  - Relevant only for popular/interactive LLMs that should always be running.
- Criteria for automated scaling, such as:
  - maximum/minimum number of concurrent requests per instance 
  - maximum/minimum aggregated token throughput per second per instance

Auto-scaling thread runs in the background every fixed period of time:

- Maintain a certain dictionary tracking:
    - time of the previous check, for calculating amount of time elapsed since the previous check, and
    - some key metric for each model for automated scaling, e.g., number of concurrent requests
    - reset the metric to 0 after each check.
- Construct a list of LLMs to scale-up and another list for scaling-down.
- Invoke KScope logic to scale-up or scale-down LLMs accordingly.
