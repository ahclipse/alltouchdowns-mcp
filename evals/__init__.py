"""The eval suite: does an LLM agent, given this server's tools, answer correctly?

Run `python -m evals --help`. The pieces:

    schema    what a case looks like, and the answer format the agent must use
    truth     computes each case's correct answer with SQL, independently of the server
    grade     compares an answer with the truth (no model involved)
    agent     one trial: a model, the live server's tools, one question
    runner    runs every trial, keeping scores and plumbing failures apart
    report    turns a run into numbers and a summary
    preflight free checks that the suite itself is sound, run on every pull request
"""
