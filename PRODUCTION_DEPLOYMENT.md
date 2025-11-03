# MicrobeLLM Web Production Deployment Guide

This guide captures the tasks required to harden and ship the `microbellm-web` service to a production-grade environment. Adapt each section to the target platform (Render, container orchestrator, VM, etc.) while keeping the underlying Flask + Socket.IO stack intact.

## 1. Prepare the Runtime Environment

- Provision a Python 3.11 runtime (system packages or container base image) with build tools for wheels (gcc, make, libffi, python3-dev, etc.).
- Install project dependencies with `pip install -r requirements.txt`; verify `gunicorn` and `eventlet` resolve properly since Socket.IO requires an async worker.
- Lock dependency versions (pip-tools, Poetry, or pinned requirements) and set up a reproducible build artifact (Docker image, immutable VM AMI, or Render build).
- Harden the OS/container: create an unprivileged service account, enable automatic security updates, and restrict outbound networking if only OpenRouter access is required.

## 2. Manage Secrets and Configuration

- Export required environment variables via the platform's secret manager:
  - `OPENROUTER_API_KEY` for inference features.
  - `MICROBELLM_DB_URL` if the database should be fetched from a remote object store on boot.
  - `PORT` injected by the hosting platform (defaults to 5000 locally).
- Ensure `MICROBELLM_SECRET_KEY` is provided via the platform's secret manager; the application now reads the Flask `SECRET_KEY` from that environment variable (falling back to a generated ephemeral value only for ad-hoc local runs). Copy `.env.local.example` to `.env.local` when testing locally.
- Optionally expose tuning knobs (thread pool limits, cache durations, rate limits) through environment variables so production adjustments do not require code edits.

## 3. Provision and Migrate the Database

- Decide whether to keep SQLite (`microbellm.db`) or migrate to a managed SQL service. SQLite is easiest for read-heavy dashboards but must reside on durable storage (persistent disk, S3/bucket download, or managed volume).
- If hosting SQLite, stage the latest database file in an object store and configure `MICROBELLM_DB_URL` so `run_web.py` can download it during release. Ensure daily backups and checksum verification.
- For multi-writer or scaling needs, plan a migration path to PostgreSQL/MySQL by refactoring the ORM/database layer (currently raw SQLite connections); validate schema compatibility and concurrency behaviour before switching.
- Seed prediction caches and research assets if the deployment should come online with precomputed results.

## 4. Build the Deployable Artifact

- Container approach: craft a Dockerfile that copies the repo, installs dependencies, downloads the database (or retrieves at runtime), and sets the entrypoint to `python run_web.py`. Run scans (Trivy/Grype) and publish to a registry.
- Render/VM approach: use `render.yaml` or system provisioning scripts to execute `pip install -r requirements.txt`, then launch `python run_web.py` on boot.
- Apply CI/CD automation (GitHub Actions, Render auto-deploy, etc.) to build the artifact on every tagged release and run the test suite (`pytest`) before promotion.

## 5. Configure the Application Server

- Serve Flask through Gunicorn with the eventlet worker class to keep WebSocket support: `gunicorn --bind 0.0.0.0:$PORT --worker-class eventlet --workers 1 --timeout 120 microbellm.web_app:app` (increase workers after measuring load).
- Terminate TLS at the platform load balancer or front proxy (nginx/Traefik) forwarding traffic to Gunicorn over HTTP.
- Enable structured logging (JSON or key-value) either through Gunicorn flags or a wrapper script; ship logs to a centralized system (CloudWatch, Loki, etc.).
- Ensure health checks probe an inexpensive endpoint (e.g., `/healthz` implemented via a lightweight Flask route) so orchestration can restart unhealthy pods.

## 6. Operational Readiness

- Run the automated test suite (`pytest`) plus smoke tests that hit key endpoints (`/`, `/api/*`) against a staging environment before promoting to production.
- Instrument observability: configure uptime monitoring, basic metrics (Gunicorn worker counts, request latency), and alerts for database fetch failures or API-rate-limit errors.
- Document runbooks for resetting stalled jobs—`reset_running_jobs_on_startup()` will mark in-flight tasks as interrupted, but operators still need manual procedures for re-queuing or inspecting anomalies.
- Establish a change-management workflow: versioned releases, rollback scripts, and incident response contacts.

## 7. Post-Deployment Tasks

- Validate that caches warm correctly and Socket.IO clients stay connected under load; tune `requests_per_second` and `max_concurrent_requests` via configuration if necessary.
- Monitor API usage against OpenRouter limits and rotate API keys securely when compromised or expiring.
- Schedule periodic regeneration of research artifacts (figures, CSV exports) so the public portal reflects the newest benchmark runs.
- Capture user metrics/feedback (privacy-compliant) to guide future optimizations.

---
Need a platform-specific checklist (Render, Kubernetes, bare metal)? Start from this guide and expand each section with concrete commands, scripts, and automation your team prefers.
