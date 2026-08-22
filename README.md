<h1 align="center">Repo Agent</h1>
<p align="center">
<a href="pyproject.toml"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge"></a> <img alt="Primary language" src="https://img.shields.io/badge/language-Python-3776AB.svg?style=for-the-badge"> <img alt="Files documented" src="https://img.shields.io/badge/files%20documented-102-informational.svg?style=for-the-badge">
</p>

---

This repository implements an automated codebase analysis and documentation generation pipeline. It ingests source code, parses it into structured representations, builds a knowledge graph, generates embeddings, and produces multiple artifacts including documentation, architecture diagrams, and code reviews. The system is designed to be modular, allowing different agents to handle specific tasks like parsing, chunking, reviewing, and exporting, all coordinated through a central state management layer.

## 🔄 Pipeline Stages

The system executes a verified sequence of stages, where certain steps depend on the completion of prior stages.

### Initialization and Scanning
The process begins with **initialization**, setting up the environment and state. This is followed by **scanning**, where the system identifies files and their contents. These initial steps prepare the raw data for the subsequent processing phases.

### Parsing and Dependency Resolution
Once files are scanned, the system moves to **parsing**. Multiple parsers (for Go, Python, Rust, TypeScript) convert source code into structured objects. This is followed by **dependency resolution**, where the system analyzes relationships between files and modules.

### Chunking and Graph Construction
The parsed data undergoes **chunking**, breaking large files into manageable units. These chunks are then processed by `chunk_doc` agents to generate document-level representations. The system constructs a **knowledge graph**, linking entities and relationships across the codebase.

### Embedding and Storage
After the graph is built, the system performs **embedding**, converting graph nodes and edges into vector representations. These vectors are stored in a **vector database** (LanceDB) for efficient retrieval and similarity search.

### Repository Understanding
The system then enters a **repository understanding** phase, where specialized agents analyze the codebase. This includes:
- **Architecture Agent**: Generates high-level architectural diagrams.
- **API Agent**: Documents and analyzes API interfaces.
- **Readme Agent**: Creates or updates project READMEs.
- **Structure Agent**: Analyzes project structure and organization.
- **Changelog Agent**: Generates changelog entries based on code changes.
- **Package Dependencies Agent**: Manages and documents package dependencies.

These agents focus on specific aspects of the repository.

### GitHub Readme and Review
Following the understanding phase, the system generates a **GitHub Readme**, likely integrating the outputs from the various agents. Finally, a **review** stage occurs, where the generated documentation and artifacts are checked against the grounding data to ensure accuracy and consistency.

## 🧩 Core Components

### Agents
The system relies on a suite of specialized agents, each responsible for a distinct part of the pipeline. These agents interact through the state management system to pass data between stages.

- **Review Agent**: Validates generated documentation against the knowledge graph and grounding data.
- **Repo Agent**: Coordinates the overall repository analysis process.
- **Vector DB Agent**: Manages interactions with the LanceDB vector store.
- **Parser Agents**: Handle syntax analysis for multiple languages (Go, Python, Rust, TypeScript).
- **Chunker Agent**: Breaks files into logical units for further processing.
- **Embedder Agent**: Generates vector embeddings for graph nodes.
- **Exporters**: Assemble and format the final outputs (JSON, etc.).

### Graph and State Management
The `graph` module provides the structural backbone of the system. It manages the lifecycle of the knowledge graph, including stages for chunking, dependency resolution, and review. The `state` module maintains the runtime state across concurrent agents and stages.

### Utilities and Helpers
Various utility modules support the core functionality:
- **Logger**: Handles logging across the system.
- **Timers**: Manages timing and progress tracking.
- **Banned Phrases**: Provides lists of prohibited terms for review.
- **File References**: Manages references to files and their contents.
- **Hashing**: Generates file hashes for integrity checks.

## 🏗️ Design Patterns

### Concurrent Execution
A key design pattern in this system is the ability to run multiple agents together. For instance, the architecture, API, readme, structure, changelog, and package dependency agents all execute during the repository understanding phase. This parallelism allows the system to generate multiple artifacts, reducing overall processing time.

### Modular Abstraction
The system uses a modular architecture where each agent is self-contained and interacts with the system through well-defined interfaces. This modularity allows for easy extension and maintenance. For example, the LLM factory pattern allows different language models to be swapped in without changing the core logic of the agents.

### Grounding and Validation
The system employs a grounding mechanism where generated content is validated against the knowledge graph and grounding data. This ensures that the documentation and reviews are accurate and consistent with the actual codebase.

## 📄 License

MIT
