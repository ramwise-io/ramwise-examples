# Microsoft Fabric CI/CD quick start

This guide describes a reusable Azure DevOps pattern for deploying Microsoft Fabric items.

## 1. Design at a glance

```text
Shared once per trusted Azure DevOps boundary
  Microsoft Entra deployment workload identity
    (app registration + tenant service principal / Enterprise application)
  Fabric API security group and tenant setting
  Workload-identity-federated Azure service connection
  Reusable deploy-fabric-steps.yml template

Per Fabric project
  DEV, TEST, and PROD workspaces
  Project variable group with TEST and PROD workspace IDs
  Project config.yml and optional parameter.yml
  Fabric item folders synchronized from DEV
  Project PROD Azure DevOps Environment and approval
  Optional project-only post-deployment steps
```

A service connection is not required per repository. Reuse one connection across trusted repositories in the same Azure DevOps project, and grant the deployment workload identity access only to the Fabric workspaces it needs. In larger organizations, split identities by trust boundary, commonly non-PROD and PROD.

### Current Microsoft Entra terminology

Microsoft increasingly uses **workload identity** as the umbrella term for non-human identities. This setup uses an **application workload identity**, not a managed identity.

| Portal or context | Term | Role in this setup |
|---|---|---|
| Microsoft Entra ID → App registrations | App registration / application object | Holds the client ID and federated credential configuration |
| Microsoft Entra ID → Enterprise applications | Enterprise application | Portal representation of the tenant-local service principal |
| Microsoft Graph, Azure RBAC, Fabric permissions | Service principal | The actual tenant identity that receives access |
| Azure DevOps | Workload identity federation service connection | Obtains short-lived tokens for that service principal |
| Microsoft Entra Workload ID | Workload identity | Broader category containing service principals and managed identities |

The most precise description is:

> Azure DevOps deploys through a workload identity federation service connection backed by a Microsoft Entra app registration and its tenant service principal.

The existing `sp-fabric-cicd` naming convention remains valid. The `sp` prefix refers to the underlying service principal. No code changes are required: Azure DevOps still exposes variables such as `servicePrincipalId` during federated authentication.

## 2. What each repository file does

### `cicd/templates/deploy-fabric-steps.yml`

The reusable deployment engine. It has no item names or business logic. It:

1. Checks out the repository.
2. Selects Python 3.12.
3. Installs the pinned `ms-fabric-cli` package.
4. Enables Fabric CLI plaintext encryption fallback for the ephemeral Linux agent.
5. Logs in with the federated token supplied by the Azure DevOps service connection.
6. Injects the target workspace ID into `config.yml` at runtime.
7. Runs `fab deploy`.
8. Logs out when the task ends.

The plaintext fallback is needed because Microsoft-hosted Linux agents do not provide a usable encrypted credential vault to Fabric CLI. Hosted agents are discarded after the job. Review this choice before using a persistent self-hosted agent.

### `fabric/<solution>/config.yml`

Project-local `fabric-cicd` configuration.

```yaml
core:
  workspace_id: "00000000-0000-0000-0000-000000000000"
  repository_directory: "."
  parameter: "parameter.yml"

publish:
  skip: false

unpublish:
  skip: true
```

- `workspace_id` is a required placeholder replaced by the pipeline.
- `repository_directory` points to the Fabric item folders.
- `parameter` enables project-specific environment replacement.
- Omitting `item_types_in_scope` deploys every supported item type discovered in the directory.
- `unpublish.skip: true` prevents automatic deletion of remote items absent from Git.

### `fabric/<solution>/parameter.yml`

Environment-specific definition replacement. Keep it empty when Fabric auto-binding or Variable Libraries already solve the dependencies.

Typical uses:

- Notebook default Lakehouse rebinding
- Warehouse or connection references
- Environment URLs
- Cross-workspace dependencies

This file stays per project because those dependencies differ.

### `ci/deploy-test.yml`

Thin TEST entry point. It supplies:

- Variable group
- Shared service connection name
- TEST workspace ID
- Project `config.yml` path

Project-specific actions may follow the shared template — for example, uploading sample data after deployment.

### `ci/deploy-prod.yml`

Thin PROD entry point implemented as an Azure DevOps deployment job. The job targets a project-specific Azure DevOps Environment, which holds the manual approval and deployment history.

### Optional project actions

Keep these outside the reusable deployment template:

- Sample or reference-data seeding
- Shortcut creation
- Pipeline or notebook execution
- Business smoke tests
- External connection provisioning
- Notifications

A project may add them after the shared deployment step or omit them entirely.

## 3. One-time tenant and Azure DevOps setup

### A. Create the deployment workload identity

Create one Microsoft Entra **app registration** for the trusted set of Fabric deployments, for example:

```text
sp-fabric-cicd
```

Creating the app registration also creates its tenant-local **service principal**, shown in the portal under **Enterprise applications**.

Use the two related objects as follows:

- Configure the federated credential under **App registrations**.
- Grant Fabric workspace access and security-group membership to the **Enterprise application / service principal**.
- Do not select a managed identity; this pattern uses an application service principal.

Create a security group such as:

```text
sg-fabric-public-api-cicd
```

Add the Enterprise application/service principal to the group.

### B. Enable Fabric API access

In the Fabric Admin portal, enable:

```text
Service principals can call Fabric public APIs
```

Scope the setting to the CI/CD security group rather than the entire tenant.

### C. Create the Azure DevOps service connection

In the Azure DevOps project:

```text
Project settings
→ Service connections
→ New service connection
→ Azure Resource Manager
→ Workload identity federation
```

Use the smallest practical Azure scope. A dedicated empty resource group is sufficient for a portfolio setup because Fabric authorization is separately controlled by Fabric workspace roles.

Recommended generic name:

```text
sc-fabric-cicd
```

Do not grant access to every pipeline unless the entire Azure DevOps project is one trusted boundary. Authorize consuming pipelines explicitly.

## 4. Setup for each Fabric project

### A. Create workspaces

Create:

```text
ws_<solution>_dev
ws_<solution>_test
ws_<solution>_prod
```

Assign the required Fabric capacity.

### B. Grant the deployment workload identity access

Add the shared Enterprise application/service principal to TEST and PROD with the Fabric workspace role required for deployment. Contributor is sufficient for the item deployment pattern used here.

DEV normally remains user-managed and Git-connected.

### C. Connect DEV to Git

Use Fabric Git integration to synchronize the DEV item definitions into the repository. The directory should resemble:

```text
fabric/<solution>/
  lh_solution.Lakehouse/
  nb_transform.Notebook/
  pl_orchestrator.DataPipeline/
  config.yml
  parameter.yml
```

The item folders in Git are the deployment inventory. Do not maintain a separate expected-items manifest.

### D. Create the variable group

Create:

```text
vg-fabric-<solution>
```

Add:

```text
FABRIC_TEST_WORKSPACE_ID
FABRIC_PROD_WORKSPACE_ID
```

These are configuration values, not credentials.

### E. Customize the thin pipelines

In `ci/deploy-test.yml` and `ci/deploy-prod.yml`, set:

- Variable group name
- Shared service connection name
- `configPath`
- PROD Azure DevOps Environment name

The service connection and environment names must be literal YAML values because Azure DevOps authorizes protected resources before runtime variables are expanded.

### F. Configure PROD approval

Create an Azure DevOps Environment:

```text
fabric-<solution>-prod
```

Open **Approvals and checks**, add an approval check, and select the approver. Approvals are configured on the Environment, not in YAML.

### G. Run deployments

1. Run the TEST pipeline.
2. Validate the Fabric items and solution behavior in TEST.
3. Run the PROD pipeline.
4. Approve the Environment check.
5. Validate the solution in PROD.

## 5. Extending the pattern

### Add Fabric artifacts

Add or synchronize another supported Fabric item folder beneath the configured `repository_directory`. No expected-items list needs updating.

### Add another environment

Create another workspace-ID variable and a thin pipeline that calls the same deployment template with a new `targetEnvironment` value.

### Host the template centrally

After two or more repositories use the same template, move it to a dedicated repository such as:

```text
fabric-cicd-templates
```

Reference a pinned tag or commit from application repositories so an untested template change cannot affect every deployment at once.

### Use separate PROD identity

For stronger organizational isolation, use:

```text
sc-fabric-cicd-nonprod
sc-fabric-cicd-prod
```

Grant each identity access only to its environment class. This is generally better than one identity per repository.

### Enable orphan deletion

Change `unpublish.skip` only after agreeing on ownership and deletion policy. With unpublish enabled, items missing from the repository can be removed from the target workspace.

### Add pull-request validation

A lightweight CI pipeline can validate YAML/JSON syntax and inspect Fabric definitions without deploying. Keep business tests separate from the generic deployment engine.

### Prefer Fabric-native bindings

Use Fabric auto-binding and Variable Libraries where supported. Keep `parameter.yml` only for dependencies that still require definition replacement.

## 6. Troubleshooting

### Encrypted-cache error

```text
An error occurred with the encrypted cache
```

The supplied template addresses this with:

```bash
fab config set encryption_fallback_enabled true
```

### HTTP 401

Check the workload identity federation service connection, the app registration's federated credential, tenant ID, and Fabric CLI login.

### HTTP 403

Check the Fabric public-API tenant setting, Enterprise application/service-principal security-group membership, and target workspace role.

### Item still points to DEV

Inspect the deployed item definition and add or correct the project-local `parameter.yml` replacement. Also check Fabric auto-binding settings.

### Fabric items deploy but data is absent

Git and `fabric-cicd` deploy item definitions, not Lakehouse files or table contents. Data must be ingested, copied, seeded, or exposed through shortcuts separately.

### New artifact is not deployed

Confirm its item type is supported by the installed Fabric CLI / `fabric-cicd` version and that the item folder is under `core.repository_directory`.

## 7. Official references

- Fabric CLI authentication: https://microsoft.github.io/fabric-cli/commands/auth/
- Fabric CLI settings: https://microsoft.github.io/fabric-cli/essentials/settings/
- `fabric-cicd` configuration deployment: https://microsoft.github.io/fabric-cicd/latest/how_to/config_deployment/
- `fabric-cicd` parameterization: https://microsoft.github.io/fabric-cicd/latest/how_to/parameterization/
- Microsoft Entra workload identities: https://learn.microsoft.com/en-us/entra/workload-id/workload-identities-overview
- Microsoft Entra app and service principal objects: https://learn.microsoft.com/en-us/entra/identity-platform/app-objects-and-service-principals
- Fabric developer tenant settings: https://learn.microsoft.com/en-us/fabric/admin/service-admin-portal-developer
- Fabric workspace roles: https://learn.microsoft.com/en-us/fabric/fundamentals/roles-workspaces
- Azure DevOps workload identity federation: https://learn.microsoft.com/en-us/azure/devops/pipelines/release/configure-workload-identity
- Azure DevOps approvals and checks: https://learn.microsoft.com/en-us/azure/devops/pipelines/process/approvals
