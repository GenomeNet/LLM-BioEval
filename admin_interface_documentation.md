# MicrobeLLM Admin Interface Documentation

This document provides a detailed overview of the MicrobeLLM admin interface, a platform designed for managing and running Large Language Model (LLM) inference jobs for microbiome research. It covers the core scientific workflow, experiment management capabilities, and the underlying technical architecture.

## 1. Overview and Architecture

The MicrobeLLM admin interface is a Flask-based web application that provides a graphical user interface (GUI) for managing the entire lifecycle of LLM-based research experiments. The platform is designed for local deployment to give researchers full control over their data and models.

The key architectural components are:
-   **Web Application (`admin_app.py`):** The main Flask application that serves the user interface and handles API requests.
-   **Processing Manager:** The core engine that manages job execution, concurrency, and rate limiting.
-   **Template and Validation System:** A system for standardizing prompts and parsing LLM outputs.
-   **Database Interface (`unified_db.py`):** An abstraction layer for all database interactions, centered around a unified SQLite database.

## 2. Core Scientific Workflow

The platform is built around a reproducible scientific workflow that consists of three main stages: prompt engineering, structured data generation and validation, and model evaluation.

### 2.1. Prompt Engineering with the Template System

The quality and consistency of the LLM's output are managed through a robust template system. This system ensures that prompts are well-defined and that the responses can be reliably parsed.

-   **System Templates:** These templates provide high-level instructions to the LLM. They define the role of the model (e.g., "You are a helpful assistant with expertise in microbiology"), the desired output format (JSON), and the specific fields and data types to be returned.
-   **User Templates:** These templates contain the specific query, which typically includes a placeholder for the species name (e.g., `{{ species_name }}`).

An experiment, or "job," is always defined by a pair of a system and a user template. This separation allows for systematic testing of different prompting strategies. The admin interface provides a `/templates` page to view and manage these templates.

### 2.2. Structured Data Generation and Validation

The templates instruct the LLM to return its response in a structured JSON format. A typical JSON response might include fields for `knowledge_group`, `gram_staining`, `motility`, and other microbial phenotypes.

After receiving a response from the LLM, the system performs the following steps:
1.  **JSON Parsing:** The raw text response is parsed into a JSON object. The system includes robust error handling to manage cases where the LLM produces malformed JSON.
2.  **Data Extraction:** The values for each phenotype are extracted from the JSON object.
3.  **Validation (Against Ground Truth):** If ground truth data is available for a species, the extracted phenotypes are compared against the ground truth values. The `/ground_truth` page in the admin interface allows for managing these ground truth datasets.

### 2.3. Model Performance Evaluation

The platform provides functionalities to evaluate the performance of different LLM models based on the validated results. The evaluation is typically based on accuracy, comparing the model's predictions for each phenotype against the ground truth data. The validation and evaluation results can be explored through various analytical pages in the web interface, allowing researchers to compare the performance of different models and prompt strategies.

## 3. Experiment Execution and Management

The platform provides comprehensive tools for managing the lifecycle of experiments, ensuring reproducibility and efficient use of resources.

### 3.1. The Processing Manager

The `ProcessingManager` class is the core component for managing inference jobs. It is responsible for:
-   **Job Queuing and Execution:** It manages a queue of jobs to be processed.
-   **Concurrency Control:** It uses a `ThreadPoolExecutor` to run multiple jobs concurrently, with a configurable maximum number of concurrent requests.
-   **Rate Limiting:** It enforces a configurable rate limit on requests made to external LLM APIs to prevent exceeding API limits.

### 3.2. Job Lifecycle Control

The web interface provides fine-grained control over job execution, which is crucial for managing long-running experiments and handling errors:
-   **Start, Pause, and Stop:** Jobs can be started, paused at any time, and resumed later. A running job can also be stopped completely.
-   **Restart:** A completed or stopped job can be restarted, which will re-queue all species for processing.
-   **Rerun Failed:** A crucial feature for handling intermittent failures. This re-queues only the species that failed or timed out during the last run, providing an efficient way to recover from transient network issues or API errors without re-running the entire job.
-   **Reparse Results:** Allows for reprocessing the raw results from a completed job with updated parsing logic.

## 4. Technical Implementation Details

This section covers the underlying technical components of the platform.

### 4.1. Database Backend

The application uses an SQLite database to store all data, including job configurations, results, and ground truth data. The `UnifiedDB` class (`microbellm/unified_db.py`) provides a high-level API for all database operations, such as creating jobs, updating results, and querying data for the dashboard. The core table is `processing_results`, which stores all information about each individual prediction.

### 4.2. Web Interface and API

The user-facing part of the platform is a web interface with pages for the main dashboard, settings, a database browser, and template management. The interface is powered by a set of JSON-based API endpoints that provide programmatic control over the system's functionalities, including job and data management.

### 4.3. Real-time Monitoring

The platform uses Flask-SocketIO to provide real-time updates to the web interface. This is crucial for monitoring the progress of jobs without needing to manually refresh the page. The server pushes updates to the client on events like `dashboard_update`, `log_message`, and `progress_update`.

### 4.4. Application Startup

The application is started by running `microbellm-admin` from the command line. The entry point is the `main()` function in `microbellm/admin_app.py`, which parses command-line arguments (e.g., `--host`, `--port`), initializes the `ProcessingManager`, and starts the web server.

## 5. Conclusion

The MicrobeLLM admin interface is a comprehensive tool for managing LLM inference jobs in a research context. It provides a user-friendly web interface for creating, monitoring, and managing jobs, with a strong emphasis on a reproducible scientific workflow. The platform's features for prompt engineering, structured data validation, and robust experiment management make it an essential tool for large-scale LLM-based microbiome research.
