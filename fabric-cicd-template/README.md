# fabric-cicd-template

A small Azure DevOps starter for deploying Microsoft Fabric items to TEST and PROD.

Companion to [Deploying Fabric Without a Debugger](https://ramwise.dev/blog/deploying-fabric-without-a-debugger/).

## What this is and isn't

The work here is **not** the deployment engine. Two Microsoft tools do that:

- [`fabric-cicd`](https://github.com/microsoft/fabric-cicd) — publishes Fabric item definitions from a folder
- [`ms-fabric-cli`](https://microsoft.github.io/fabric-cli/) — the `fab` command line, pinned here to `1.6.1`

This is just the wiring around them: one reusable pipeline template, one
release pipeline that promotes TEST → PROD, and a config layout that survives
having three environments.

## Layout

```text
cicd/templates/deploy-fabric-steps.yml   the whole engine — project-agnostic
ci/release.yml                           TEST → PROD promotion, two stages
fabric/solution/config.yml               fabric-cicd config
fabric/solution/parameter.yml            environment replacements (starts empty)
docs/FABRIC_CICD_QUICKSTART.md           full setup, file by file
```

## The decisions worth copying

- **The shared template knows nothing about your project.** No item names, no
  business rules, no data seeding. Everything project-specific lives in the
  consuming repo, so one template can serve every project.
- **Workspace GUIDs never enter Git.** `config.yml` carries a placeholder that
  the pipeline overrides at runtime from a variable group.
- **The item folders are the inventory.** `fabric-cicd` deploys every supported
  item folder it finds. There is no second manifest to drift out of sync.
- **`unpublish.skip: true`.** The deployer is not allowed to delete a remote
  item just because it's missing from Git. Turn this on only after you've
  agreed who owns deletion.
- **Both stages are deployment jobs with an Environment.** That's where the
  approval and the deployment history live — not in YAML. Only PROD carries a
  check; TEST's environment exists for history and a consistent structure.

## Three things that cost me time

- **Hosted agents have no credential vault.** Fabric CLI wants an encrypted
  cache and fails with `An error occurred with the encrypted cache`. Hence
  `fab config set encryption_fallback_enabled true` — acceptable because the
  agent is ephemeral. Reconsider it on a persistent self-hosted agent.
- **Deployment jobs don't check out your repo.** Normal jobs do. The template
  has an explicit `checkout: self` so it works in both.
- **Service connection and Environment names must be literal YAML.** Azure
  DevOps authorizes protected resources *before* runtime variables expand, so
  `$(MY_CONNECTION)` fails in a way that doesn't obviously say why.

## Using it

1. Copy this folder into your project repo and replace the `<project>` placeholders.
2. Put your DEV-synchronized Fabric item folders under `fabric/solution/`.
3. Create `vg-fabric-<project>` with `FABRIC_TEST_WORKSPACE_ID` and `FABRIC_PROD_WORKSPACE_ID`.
4. Point a workload-identity-federation service connection at it, and give that
   service principal Contributor on TEST and PROD.
5. Create both Environments — `fabric-<project>-test` and `fabric-<project>-prod`
   — and add an approval check on the PROD one only.
6. Create one pipeline pointing at `ci/release.yml`. That's the whole thing:
   TEST runs, then PROD waits for approval, both from a single commit.

The [quick start](docs/FABRIC_CICD_QUICKSTART.md) has the whole thing, including
the Entra identity setup and a troubleshooting list.

## Caveat

This is a portfolio-scale setup that deploys reliably for me. It is not a
hardened enterprise reference. In a larger org you'd split identities by trust
boundary, host the template in its own repo, and pin consumers to a tag.
