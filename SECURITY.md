# Security Policy

## Reporting a vulnerability

Report privately through [GitHub's advisory form](https://github.com/brekkylab/ailoy/security/advisories/new),
or by email to oceanjoon@brekkylab.com. Please do not open a public issue for a suspected
vulnerability, and do not include a working exploit or a real credential in any report.

Include the language you used Ailoy from, the package version, what you ran, what you expected,
and what happened. A reproduction that needs no provider key, such as one against a registered
fake model as the tests do, is ideal.

We aim to acknowledge a report within three working days and to describe the fix or the
disagreement within fourteen. Fixes ship in a normal release, credited to the reporter unless
they ask otherwise.

Only the latest release on crates.io, PyPI and npm is supported.

## What counts

Ailoy gives an agent a computer of its own, so the interesting failures are the ones that break
the boundaries it claims to enforce:

- **The console reaching the host beyond what was mounted.** A command, a tool or a file
  operation in the console that reads or writes a host path outside the session's mounts, writes
  under a mount declared read-only, or reaches the network from a session built with
  `network(false)`.
- **A credential reaching somewhere it was not given to.** A provider key, or anything else from
  the host environment, that ends up inside the console, in a tool result, in a message sent to
  a model, or in a log.
- **Code execution from data.** A model's response, a tool result or a skill file that runs on
  the host, rather than being handed to the console or to the model.
- **One agent reaching another's state** through a shared console, memory or registry when the
  API says they are separate.

## What does not

- **A model doing something unwise inside its console.** The console is the agent's computer,
  and what it installs, deletes or runs there is the agent's business. The boundary is the host.
- **Binding and exposure are the operator's call.** The console server listens for the process
  that started it; putting it or an agent behind a public interface is a deployment decision.
- **The keys in the tests are not secrets.** The registered fake models in the test suites
  answer from an in-process server, and the tokens they take authenticate nothing.
- **virtx itself.** A boundary that fails inside the micro-VM or its server belongs to
  [virtx](https://github.com/brekkylab/virtx/security), which Ailoy builds on; report it there,
  or here if you are unsure and we will pass it on.
