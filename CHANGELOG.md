# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-06-26

### Added

- Validation / correctness benchmark suite: GA + NSGA-II parity vs pymoo and DEAP
  across single- and multi-objective problems, with committed 30-seed results, a
  rendered report, and a published Validation docs page.

### Notes

- No breaking changes. The optimizer API and core are unchanged since 0.1.0; the
  benchmark numbers therefore remain valid for 0.2.0.

## [0.1.0] - 2026-06-25

### Added

- Initial public release.
- Pure-numpy genetic algorithm framework for single-objective optimization (`ga`).
- NSGA-II multi-objective optimization (`nsga2`) with non-dominated sorting and crowding-distance.
- Pluggable selection strategies (tournament, roulette, crowded-tournament) and survival strategies (elitist, truncation, NSGA-II) via a registry.
- SBX crossover and polynomial-mutation operators with single-seed reproducibility.
- Optional parallel evaluation via joblib (`n_workers`).
- Typed public API shipping a `py.typed` marker (PEP 561).

[Unreleased]: https://github.com/hydrosolutions/ctrl-freak/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/hydrosolutions/ctrl-freak/releases/tag/v0.2.0
[0.1.0]: https://github.com/hydrosolutions/ctrl-freak/releases/tag/v0.1.0
