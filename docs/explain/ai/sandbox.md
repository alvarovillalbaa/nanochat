# Code Execution Reliability Guard

`nanochat/execution.py` provides a subprocess reliability guard for HumanEval.
It is **not a security sandbox** and must not be used to execute adversarial or
otherwise untrusted code on an unprotected host.

## Overview

The evaluator runs generated Python in a child process, captures stdout/stderr,
and returns a structured result.

## Security & Safety Mechanisms

The implementation includes best-effort guards against accidental damage and
infinite loops:

1.  **Child process**: Each snippet runs in a separate `multiprocessing.Process`, which lets the parent kill a hung execution. A child process is not a security boundary.
2.  **Timeouts**: Execution is strictly limited by a timeout (default 5 seconds). If the code takes longer, the process is killed.
3.  **Memory Limits**: Uses `resource.setrlimit` to enforce a maximum memory usage (default 256MB) to prevent OOM attacks.
4.  **Function Disabling**: Dangerous functions are monkey-patched to `None` to prevent their use:
    *   File system operations: `os.remove`, `shutil.rmtree`, `os.rename`, etc.
    *   Process operations: `os.system`, `subprocess.Popen`, `os.fork`, `os.kill`.
    *   Network/System: `os.setuid`, `os.chroot`.
5.  **Temporary Directory**: Code runs in a temporary directory that is automatically cleaned up after execution.

## Limitations

*   **Network access is not blocked.**
*   Python dynamic features can bypass monkey patches.
*   There is no seccomp, container, virtual machine, or kernel security boundary.
*   Run the evaluator inside a real external sandbox when code is untrusted.

## Usage

The `execute_code` function is the entry point for controlled evaluation:

```python
from nanochat.execution import execute_code

result = execute_code("print('hello world')")
if result.success:
    print(result.stdout)
```
