# Agent session transcripts

Raw session logs from the CodeBuddy Code CLI that built this repository, kept as
a record of how the design decisions were reached. One JSONL file per session;
each line is a typed record (`message`, `reasoning`, `function_call`,
`function_call_result`, `file-history-snapshot`, ...).

These are unedited snapshots. They contain absolute paths from the machine the
project was developed on, the agent's internal reasoning, and the full text of
files that were read or written. They are **not** part of the build and nothing
in the repository references them.
